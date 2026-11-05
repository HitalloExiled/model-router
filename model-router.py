#!/usr/bin/env python3
"""Model router: Ollama-style load/unload across llama.cpp serve profiles.

Listens on MODEL_ROUTER_PORT (default 18089) and forwards to a single
llama.cpp backend on MODEL_ROUTER_BACKEND_PORT (default 8080). The requested
OpenAI model name picks the backend profile:

  qwen3.8-gguf-*    -> gguf-stack/serve-llama.sh    (Qwen3.8 27B, GGUF)
  bonsai-2-27b-*    -> bonsai-stack/serve-bonsai.sh (Bonsai 2 27B, ternary)

Any other model name reuses whatever backend is running (the server accepts
arbitrary model strings). Switching profiles stops the running container and
boots the wanted one (~60s cold start); after IDLE_SEC seconds without
requests the backend is stopped to free VRAM. The default is 6 h: the GPU has
no other tenant, and a 15-minute unload threw away a warm 100K-token context
on every coffee break (a ~63 s re-prefill on the next turn).

Env:
  MODEL_ROUTER_PORT / MODEL_ROUTER_BACKEND_PORT / MODEL_ROUTER_IDLE_SEC
    (default 18089 / 8080 / 21600)
  MODEL_ROUTER_LLAMA_SERVE_SH / MODEL_ROUTER_BONSAI_SERVE_SH (launchers)
  NINFER_LLAMA_IMAGE / NINFER_BONSAI_IMAGE (shared with the serve scripts)
  MODEL_ROUTER_DIGEST=1 / MODEL_ROUTER_DUMP=<dir> (prefix-cache forensics)
  MODEL_ROUTER_PREFILL_GATE=0 (disable the admission gate entirely)
  MODEL_ROUTER_PREFILL_GATE_ENFORCE=0 (diagnose only; phase A is enforced)
  MODEL_ROUTER_PREFILL_GATE_TIMEOUT / _POLL / _MIN / _COLD_MIN
  MODEL_ROUTER_PREFILL_GATE_COLD=1 (also hold cold prompts behind a decoder)
  MODEL_ROUTER_PREFILL_GATE_DIAG=0 (silence gate diagnostics)
  MODEL_ROUTER_PREFILL_GATE_DIAG_VERBOSE=1 (log every admission, not just holds)
  MODEL_ROUTER_SLOT_PERSIST=0 (do not save/restore warm slots across a boot)
  MODEL_ROUTER_SLOT_DIR / _SAVE_MIN / _DISK_MIB / _IO_TIMEOUT
"""
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))


def stack_dir(name):
    """Locate a stack checkout: inside the router repo (git submodule) first,
    then as a sibling checkout (~/Projetos/My/<name>)."""
    inside = os.path.join(ROUTER_DIR, name)
    return inside if os.path.isdir(inside) else os.path.join(
        os.path.dirname(ROUTER_DIR), name)


SERVE_LLAMA = os.environ.get(
    "MODEL_ROUTER_LLAMA_SERVE_SH",
    os.path.join(stack_dir("gguf-stack"), "serve-llama.sh"))
SERVE_BONSAI = os.environ.get(
    "MODEL_ROUTER_BONSAI_SERVE_SH",
    os.path.join(stack_dir("bonsai-stack"), "serve-bonsai.sh"))
# llama.cpp stack (GGUF). Flags live in gguf-stack/serve-llama.sh; the router
# only selects profile -> argv here. Image names are shared with the serve
# scripts (which own the NINFER_* env names): the router reads them to
# recognize a running container, the scripts to pick the image to run.
LLAMA_IMAGE = os.environ.get(
    "NINFER_LLAMA_IMAGE",
    "ghcr.io/ggml-org/llama.cpp:server-cuda-b12110")
GGUF_FILES = {
    "gguf-speed": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
    "gguf-range": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    "gguf-max": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    "gguf-duo": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    "gguf-duo128": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    "gguf-quad": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    "gguf-vision": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
    "gguf-single192": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
}
# Bonsai 2 27B (ternary) stack: PrismML fork of llama.cpp. Stock llama.cpp
# cannot run these packs, so this is a THIRD image — still one backend at a
# time: whichever model is requested wins and the resident one is stopped.
BONSAI_IMAGE = os.environ.get(
    "NINFER_BONSAI_IMAGE", "bonsai-llama-4090:sm89")
BONSAI_FILES = {
    "bonsai-max": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-duo256": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-trio256": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-trio200": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-duo128": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-quad128": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-penta128": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-hexa128": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-octo64": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "bonsai-vision": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
}
# Boot log lives under ~/.local/share (NOT /tmp): the router may run in a
# mount namespace with a private /tmp, which would hide the log from shells.
LOG_DIR = os.path.join(os.path.expanduser("~"), ".local", "share", "model-router")
BOOT_LOG = os.path.join(LOG_DIR, "backend-boot.log")
ROUTER_PORT = int(os.environ.get("MODEL_ROUTER_PORT", "18089"))
BACKEND_PORT = int(os.environ.get("MODEL_ROUTER_BACKEND_PORT", "8080"))
IDLE_SEC = int(os.environ.get("MODEL_ROUTER_IDLE_SEC", "21600"))
BOOT_TIMEOUT = 420

# Slot persistence. llama-server can dump a slot's KV cache to disk and load
# it back into a fresh process; both serve scripts wire `--slot-save-path` and
# a bind mount. Without it a tier switch or an idle unload throws away a warm
# 100K-token context and pays a full re-prefill (measured 63.2 s for 115,491
# tokens on bonsai-trio200, ~1,830 tok/s). Restore only works for the same
# weights file - the server validates the state against the loaded model - so
# the file name carries a hash of the weights basename: a Qwen dump is never
# handed to Bonsai, and tiers that share a file (gguf-range/max/duo/quad,
# every Bonsai) can hand a warm slot to each other.
#
# Since Oct 2026 the save file also carries the slot's context checkpoints
# (upstream PR #26004). Hybrid models (Bonsai, Qwen3.8) need them: without the
# appendix the server loads the KV and then throws it away ("forcing full
# prompt re-processing due to lack of cache data"). A checkpoint is ~150 MiB,
# so a long slot dump can be several GiB larger than the raw KV; the disk cap
# below is sized for two full dumps.
SLOT_PERSIST = os.environ.get("MODEL_ROUTER_SLOT_PERSIST", "1") != "0"
SLOT_DIR = os.environ.get("MODEL_ROUTER_SLOT_DIR",
                          os.path.join(ROUTER_DIR, "slot-cache"))
# Below this many tokens a re-prefill is cheaper than touching the disk.
SLOT_SAVE_MIN = int(os.environ.get("MODEL_ROUTER_SLOT_SAVE_MIN", "2048"))
# Hard cap for the cache dir; oldest files are pruned first. A 100K Bonsai
# slot is ~2.5 GiB of KV + checkpoints, a 96K Qwen q8_0 slot ~14 GiB.
SLOT_DISK_MIB = int(os.environ.get("MODEL_ROUTER_SLOT_DISK_MIB", "32768"))
SLOT_IO_TIMEOUT = float(os.environ.get("MODEL_ROUTER_SLOT_IO_TIMEOUT", "300"))

PROFILES = {"gguf-speed", "gguf-range", "gguf-max", "gguf-duo", "gguf-duo128",
            "gguf-quad", "gguf-vision", "gguf-single192",
            "bonsai-max", "bonsai-duo256", "bonsai-trio256", "bonsai-trio200",
            "bonsai-duo128",
            "bonsai-quad128", "bonsai-penta128", "bonsai-hexa128",
            "bonsai-octo64", "bonsai-vision"}
# Per-slot window (a client sees one slot, not the whole pool).
PROFILE_WINDOWS = {"gguf-speed": 65536, "gguf-range": 98304,
                   "gguf-max": 262144, "gguf-duo": 98304,
                   "gguf-duo128": 131072, "gguf-quad": 65536,
                   "gguf-vision": 65536, "gguf-single192": 196608,
                   "bonsai-max": 262144, "bonsai-duo256": 262144,
                   "bonsai-trio256": 262144, "bonsai-trio200": 204800,
                   "bonsai-duo128": 131072,
                   "bonsai-quad128": 131072, "bonsai-penta128": 131072,
                   "bonsai-hexa128": 131072, "bonsai-octo64": 65536,
                   "bonsai-vision": 65536}
# Parallel slots each profile boots with. Reusing a 1-slot backend for a
# 4-slot tier is what turns concurrency into a queue, so reuse consults this.
PROFILE_SLOTS = {"gguf-speed": 1, "gguf-range": 1, "gguf-max": 1,
                 "gguf-duo": 2, "gguf-duo128": 2, "gguf-quad": 4,
                 "gguf-vision": 1, "gguf-single192": 1,
                 "bonsai-max": 1, "bonsai-duo256": 2, "bonsai-trio256": 3,
                 "bonsai-trio200": 3,
                 "bonsai-duo128": 2, "bonsai-quad128": 4,
                 "bonsai-penta128": 5, "bonsai-hexa128": 6,
                 "bonsai-octo64": 8, "bonsai-vision": 1}

# Measured decode throughput in tok/s (2026-11-05). The MTP draft tiers are
# ~3x base Qwen; the Bonsai ternary is ~2x Qwen and its single-slot max is the
# fastest Bonsai. Values for tiers not benchmarked directly are estimates; the
# router only uses the relative order. Decode speed is what dominates `write`
# and `edit` calls: the file text (and the edit's old/new strings) is model
# output, generated token by token.
PROFILE_SPEED = {
    "gguf-speed": 97, "gguf-single192": 97,
    "gguf-range": 33, "gguf-max": 33, "gguf-duo": 33, "gguf-duo128": 33,
    "gguf-quad": 33, "gguf-vision": 33,
    "bonsai-max": 81, "bonsai-duo256": 66, "bonsai-trio256": 66,
    "bonsai-trio200": 66, "bonsai-duo128": 66, "bonsai-quad128": 66,
    "bonsai-penta128": 66, "bonsai-hexa128": 66, "bonsai-octo64": 66,
    "bonsai-vision": 66,
}
# A tier switch costs a boot and, across weights groups, a full re-prefill;
# inside a group the slot dump restores the KV. Switch to the requested tier
# only when it is this much faster: same group 1.25x, base <-> MTP 1.5x.
SPEED_SWITCH_RATIO = 1.25
SPEED_SWITCH_CROSS = 1.5
# Every GGUF/Bonsai backend CAN carry mmproj (lazy load), so media never
# forces a tier switch.
VISION_CAPABLE = {"gguf-speed", "gguf-range", "gguf-max",
                  "gguf-duo", "gguf-duo128", "gguf-quad", "gguf-vision",
                  "gguf-single192",
                  "bonsai-max", "bonsai-duo256", "bonsai-trio256",
                  "bonsai-trio200", "bonsai-duo128", "bonsai-quad128",
                  "bonsai-penta128", "bonsai-hexa128", "bonsai-octo64",
                  "bonsai-vision"}
GGUF_ALL = {"gguf-speed", "gguf-range", "gguf-max",
            "gguf-duo", "gguf-duo128", "gguf-quad", "gguf-vision",
            "gguf-single192"}
BONSAI_ALL = {"bonsai-max", "bonsai-duo256", "bonsai-trio256",
              "bonsai-trio200", "bonsai-duo128", "bonsai-quad128",
              "bonsai-penta128", "bonsai-hexa128", "bonsai-octo64",
              "bonsai-vision"}
# Everything served by a llama.cpp-family image: budgets apply here.
LLAMA_ALL = GGUF_ALL | BONSAI_ALL
# Reuse the resident backend only when it has at least the requested slots.
# Set MODEL_ROUTER_RESPECT_SLOTS=0 to restore blind tier reuse.
RESPECT_SLOTS = os.environ.get("MODEL_ROUTER_RESPECT_SLOTS", "1") != "0"

# Admission-balance gate for mixed decode/prefill workloads. Deliberately
# router-side: the vendored llama.cpp is never patched.
#
# Phase A ("hold an arrival while another slot is mid-prefill") is enforced by
# default. Measured on trio200 with two real concurrent ~10.7k-token cold
# prefills: the in-flight request finished in 3.7 s instead of 7.6 s, while the
# queued one went 8.0 s -> 8.5 s end to end. Total GPU work is unchanged; the
# interleaved case just burns throughput on mixed-sequence attention.
#
# Phase B ("a cold prompt must not prefill while a slot decodes") is off by
# default (`_COLD=0`). With a long-lived decoder - the agent session itself -
# it waited out the whole timeout and then stalled the same decoder anyway, so
# it bought no throughput, only latency.
PREFILL_GATE = os.environ.get("MODEL_ROUTER_PREFILL_GATE", "1") != "0"
PREFILL_GATE_ENFORCE = os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_ENFORCE", "1") != "0"
PREFILL_GATE_TIMEOUT = float(os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_TIMEOUT", "10"))
PREFILL_GATE_POLL = float(os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_POLL", "0.5"))
# A slot only counts as prefilling when it is working on at least this many
# tokens of its own prompt. Below that a prefill lasts a single batch and
# holding an arrival behind it would only add poll-interval latency.
PREFILL_GATE_MIN = int(os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_MIN", "1024"))
PREFILL_GATE_COLD_MIN = int(os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_COLD_MIN", "8192"))
# Phase B is opt-in: set _COLD=1 to also hold cold prompts behind a decoder.
PREFILL_GATE_COLD = os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_COLD", "0") != "0"
PREFILL_PREFIX_TTL = float(os.environ.get(
    "MODEL_ROUTER_PREFILL_PREFIX_TTL", "3600"))
PREFILL_PREFIX_MAX = int(os.environ.get(
    "MODEL_ROUTER_PREFILL_PREFIX_MAX", "4096"))
# Quiet by default: log only the decisions that actually hold a request.
PREFILL_GATE_DIAG = os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_DIAG", "1") != "0"
PREFILL_GATE_DIAG_VERBOSE = os.environ.get(
    "MODEL_ROUTER_PREFILL_GATE_DIAG_VERBOSE", "0") != "0"

# Backend families. Reuse never crosses families: answering a qwen3.8-gguf-*
# request from a resident Bonsai backend (or the reverse) returns a different
# model than the client asked for, which is a correctness bug, not a cache
# win. Inside a family every tier serves the same weights, so the wider tier
# can absorb the request and keep the prefix cache.


def family(profile):
    if profile in BONSAI_ALL:
        return "bonsai"
    if profile in GGUF_ALL:
        return "gguf"
    return None


def effective_profile(wanted, current, est_tokens, need_vision):
    """Prefer the running backend, unless the requested tier is much faster.

    A switch costs a boot and, across weights groups, a full re-prefill; inside
    a group the slot dump restores the KV. Keep the resident when it can serve
    and is not materially slower. Switch to the requested tier when it can
    serve and is faster by SPEED_SWITCH_RATIO (same weights group) or
    SPEED_SWITCH_CROSS (Qwen base <-> MTP). Without this, a `single192`
    request is served by the resident `max` at ~33 tok/s instead of the MTP's
    ~97, and every `write`/`edit` pays 3x. The rule is profile-agnostic:
    Qwen/Bonsai, base/MTP, 1..8 slots, text/vision.
    """
    if current in PROFILE_WINDOWS:
        # Every GGUF/Bonsai backend carries mmproj: media never forces a
        # switch off them.
        if need_vision and current not in VISION_CAPABLE:
            return wanted
        if family(current) != family(wanted):
            return wanted
        enough_slots = (not RESPECT_SLOTS
                        or PROFILE_SLOTS.get(current, 1)
                        >= PROFILE_SLOTS.get(wanted, 1))
        if enough_slots and est_tokens < 0.9 * PROFILE_WINDOWS[current]:
            # The requested tier can only justify a switch if it also fits.
            if est_tokens < 0.9 * PROFILE_WINDOWS.get(wanted, 0):
                cur_speed = PROFILE_SPEED.get(current, 0)
                want_speed = PROFILE_SPEED.get(wanted, 0)
                same_group = (PROFILE_GROUP.get(current)
                              == PROFILE_GROUP.get(wanted))
                ratio = (SPEED_SWITCH_RATIO if same_group
                         else SPEED_SWITCH_CROSS)
                if want_speed > cur_speed * ratio:
                    return wanted
            return current
    return wanted


lock = threading.Lock()
state = {"profile": None, "profile_mmproj": False, "busy": False,
         "last_used": 0.0, "active": 0,
         "current_model": None, "current_profile": None}
# Admission-gate bookkeeping has its own lock: `lock` is held across a backend
# boot (tens of seconds) and a request must never queue behind that just to
# ask whether its prompt is warm.
prefix_lock = threading.Lock()
# Prefixes admitted recently.  An identical body except for its final message
# is a continuation and should not be treated as a cold re-prefill.
recent_prefixes = OrderedDict()


def log(*a):
    print(f"[router {time.strftime('%H:%M:%S')}]", *a, flush=True)


def sh(*cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except Exception as e:
        return 1, str(e)


def backend_containers():
    """Every running backend container (llama.cpp or Bonsai), newest first."""
    rc, out = sh("docker", "ps", "--format", "{{.ID}} {{.Image}}")
    if rc != 0:
        return []
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in (LLAMA_IMAGE, BONSAI_IMAGE):
            found.append(parts[0])
    return found


def backend_container():
    """Newest backend container from any stack (llama or bonsai)."""
    found = backend_containers()
    return found[0] if found else None


def backend_healthy():
    """True when something answers /health on the backend port.

    `docker ps` can transiently come back empty (daemon busy, container
    mid-teardown). Treating that as "nothing is running" reboots a healthy
    backend and races a second container; the port probe is authoritative.
    """
    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=3)
        conn.request("GET", "/health")
        return conn.getresponse().status == 200
    except Exception:
        return False


def slot_weights_key(profile):
    """Stable per-weights-file key: the unit a slot dump can be restored into."""
    name = GGUF_FILES.get(profile) or BONSAI_FILES.get(profile) or profile
    return f"{family(profile) or 'x'}-{hashlib.sha1(name.encode()).hexdigest()[:10]}"


# Tiers that share a weights file share slot dumps: a switch inside a group
# restores the KV, across groups it re-prefills. Base Qwen and MTP Qwen are
# different groups; every Bonsai tier is one group.
PROFILE_GROUP = {p: slot_weights_key(p) for p in PROFILES}


def slot_file_name(profile, slot_id):
    """Cache file for one slot. Safe for `fs_validate_filename` on the server
    (no separators, no '..', < 255 chars) and bound to the weights file."""
    return f"{slot_weights_key(profile)}-slot{int(slot_id)}.bin"


def slot_restore_candidates(profile, names):
    """(slot id, filename) pairs to restore into a freshly booted profile.

    Only slots the resident profile actually has are returned: the server
    wraps an out-of-range id_slot around the pool, so a two-slot dump restored
    into a one-slot backend would silently land on slot 0 instead of being
    rejected. Tiers that share a weights file share the dumps, which is what
    makes a warm slot survive a range -> max switch.
    """
    if profile not in PROFILE_SLOTS:
        return []
    n_slots = PROFILE_SLOTS[profile]
    wanted = {slot_file_name(profile, i): i for i in range(n_slots)}
    return [(wanted[name], name) for name in dict.fromkeys(names)
            if name in wanted]


def slots_to_save(phases):
    """Slot ids worth persisting before a stop: idle and carrying a prompt."""
    if not phases:
        return []
    return sorted(int(s["id"]) for s in phases.get("slots", [])
                  if not s.get("proc")
                  and int(s.get("prompt_len") or 0) >= SLOT_SAVE_MIN)


def slot_action(action, slot_id, filename):
    """POST /slots/<id>?action=save|restore on the backend. (ok, payload)."""
    body = json.dumps({"filename": filename}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT,
                                      timeout=SLOT_IO_TIMEOUT)
    try:
        conn.request("POST", f"/slots/{int(slot_id)}?action={action}",
                     body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status == 200, resp.read(65536) or b""
    finally:
        conn.close()


def prune_slot_cache(keep=()):
    """Cap the cache dir, oldest first. Unlinking needs write on the directory,
    not on the file, so root-owned dumps written by the container are fine."""
    try:
        files = [os.path.join(SLOT_DIR, f) for f in os.listdir(SLOT_DIR)
                 if f.endswith(".bin")]
        sizes = {f: os.path.getsize(f) for f in files if os.path.isfile(f)}
    except OSError:
        return
    total = sum(sizes.values())
    cap = SLOT_DISK_MIB * 1024 * 1024
    if total <= cap:
        return
    for path in sorted(sizes, key=os.path.getmtime):
        if total <= cap:
            break
        if os.path.basename(path) in keep:
            continue
        try:
            os.unlink(path)
            total -= sizes[path]
            log(f"slot cache pruned {os.path.basename(path)}")
        except OSError:
            pass


def save_slot_cache():
    """Persist warm slots before the backend is stopped (best effort).

    Any failure only costs the re-prefill that already happens today, so this
    never blocks or delays the stop beyond the per-file I/O timeout. A save is
    skipped when the disk cannot take another dump: filling the volume would
    break every later boot, which is far worse than one cold prefill.
    """
    if not SLOT_PERSIST:
        return
    try:
        profile = state["profile"]
        phases = backend_slot_state()
        slots = slots_to_save(phases)
        if not slots or profile not in PROFILES:
            return
        os.makedirs(SLOT_DIR, exist_ok=True)
        free = shutil.disk_usage(SLOT_DIR).free
        if free < 8 * 1024 ** 3:
            log(f"slot save skipped: only {free // 1024 ** 3} GiB free")
            return
        kept = set()
        for sid in slots:
            name = slot_file_name(profile, sid)
            t0 = time.monotonic()
            ok, raw = slot_action("save", sid, name)
            if not ok:
                log(f"slot {sid} save failed: {raw[:160]!r}")
                continue
            kept.add(name)
            path = os.path.join(SLOT_DIR, name)
            mib = os.path.getsize(path) / 1048576 if os.path.exists(path) else 0
            log(f"slot {sid} saved to {name} ({mib:.0f} MiB, "
                f"{time.monotonic() - t0:.1f}s)")
        prune_slot_cache(kept)
    except Exception as e:  # the cache must never get in the way of a stop
        log(f"slot save skipped: {e!r}")


def restore_slot_cache(profile):
    """Load persisted slots into the freshly booted backend (best effort)."""
    if not SLOT_PERSIST or profile not in PROFILES:
        return
    try:
        names = sorted(os.listdir(SLOT_DIR))
    except OSError:
        return
    try:
        for sid, name in slot_restore_candidates(profile, names):
            t0 = time.monotonic()
            ok, raw = slot_action("restore", sid, name)
            if ok:
                log(f"slot {sid} restored from {name} "
                    f"({time.monotonic() - t0:.1f}s)")
            else:
                log(f"slot {sid} restore failed: {raw[:160]!r}")
    except Exception as e:
        log(f"slot restore skipped: {e!r}")


def stop_backend():
    """Stop EVERY backend container: one model up means the other goes down.

    Stopping only the newest would leave a second resident backend holding
    VRAM after a manual launch, so the pre-flight gate must see a clean card.

    Warm slots are saved first: a restart otherwise costs a full re-prefill
    (~63 s at 115K tokens). The save is best effort and skipped entirely when
    the backend is already gone.
    """
    if backend_healthy():
        save_slot_cache()
    ids = backend_containers()
    if ids:
        log("stopping backend", *ids)
        sh("docker", "stop", *ids, timeout=120)
    state["profile"] = None
    state["profile_mmproj"] = False
    with prefix_lock:
        recent_prefixes.clear()


def launch_command(profile, want_mmproj=False):
    """Argv to boot a profile on any stack. None = not wired yet."""
    if profile in BONSAI_FILES:
        script = SERVE_BONSAI
        if not os.path.isfile(script):
            return None
        # serve-bonsai profiles are max|duo256|trio256|duo128|quad128|octo64|vision.
        args = [script, "--profile", profile.removeprefix("bonsai-"),
                "--port", str(BACKEND_PORT)]
        if want_mmproj:
            args += ["--mmproj", "on"]
        return args
    if profile in GGUF_FILES:
        script = SERVE_LLAMA
        if not os.path.isfile(script):
            return None
        # serve-llama profiles are speed|range|max (gguf- prefix is routing only).
        args = [script, "--profile", profile.removeprefix("gguf-"),
                "--port", str(BACKEND_PORT)]
        if want_mmproj:
            args.append("--mmproj")
            args.append("on")
        return args
    return None


def start_backend(profile, want_mmproj=False):
    env = dict(os.environ)
    # Both serve scripts mount this dir and pass --slot-save-path /slot-cache
    # so a warm slot can survive the restart (see restore_slot_cache).
    env["NINFER_SLOT_CACHE_DIR"] = SLOT_DIR
    log(f"starting profile {profile} want_mmproj={want_mmproj}")
    # Serve output goes to a file, never DEVNULL: a dead launcher must leave
    # evidence (exit code + tail) instead of a silent 7-minute hang.
    os.makedirs(LOG_DIR, exist_ok=True)
    boot_log = open(BOOT_LOG, "ab", buffering=0)
    argv = launch_command(profile, want_mmproj)
    if argv is None:
        log(f"profile {profile} has no launcher wired yet")
        try:
            boot_log.close()
        except Exception:
            pass
        return False
    proc = subprocess.Popen(
        argv,
        stdout=boot_log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=env, start_new_session=True,
    )
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT:
        if proc.poll() is not None:
            log(f"launcher for {profile} exited rc={proc.returncode} "
                f"(see {BOOT_LOG}); aborting boot")
            try:
                boot_log.close()
            except Exception:
                pass
            return False
        try:
            conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=5)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                state["profile"] = profile
                state["profile_mmproj"] = want_mmproj
                log(f"profile {profile} listening")
                restore_slot_cache(profile)
                return True
        except Exception:
            pass
        time.sleep(3)
    log(f"profile {profile} failed to listen in {BOOT_TIMEOUT}s")
    return False


EFFORT_SUFFIXES = ("none", "low", "medium", "high", "xhigh", "minimal")

# Backend family -> {tier: (template reasoning_effort, configured budget)}.
#
# The template only accepts xhigh|medium|low. Measured 2026-11-04 on a
# 9-prompt battery (Qwen b12110 / Bonsai prism-b12128):
#   - Real agent system prompt (13 KB OpenCode prompt + AGENTS.md + toolbelt):
#     the Qwen few-shot PREPENDED still wins (100% including the needle, median
#     58 vs 73 old, realistic 1,074 vs 1,890; the example's final line has no
#     label, so code tasks stay bare); appending the style or using the "think
#     briefly" text dropped to 88% (the needle). Qwen medium uses the same
#     few-shot now, with the larger budget doing the distinction.
#     The Bonsai is the opposite: with the real system the no-style medium is
#     the best realistic (1,885 tokens / 22.2 s vs 2,575 / 29 s with the
#     few-shot); all variants scored 100%. Bonsai keeps no style.
#   - `low` caps at 1,536 so it really is the lighter tier: the Qwen realistic
#     natural is 1,631, the Bonsai 1,885; the cap forces the transition and
#     leaves the answer. `medium` keeps the larger configured budget.
EFFORT_POLICY = {
    "gguf": {
        "none": ("none", None),
        "low": ("low", 1536),
        "medium": ("medium", 3072),
        "high": ("xhigh", 8192),
        "xhigh": ("xhigh", None),
    },
    "bonsai": {
        "none": ("none", None),
        "low": ("medium", 1536),
        "medium": ("medium", 6144),
        "high": ("xhigh", 8192),
        "xhigh": ("xhigh", None),
    },
}

# One concise example beats an abstract instruction: the template's own low
# sentence was not enough for Qwen. It is prepended to the client system. The
# Bonsai has no style: its own prompt is already concise, and the example made
# the open-ended task hit the cap.
FEWSHOT_STYLE = (
    "Example of the desired style:\n"
    "Question: What is 17*24?\n"
    "17*24 = 17*20 + 17*4 = 408.\n"
    "408\n\n"
    "Solve the next task in the same concise style: at most one sentence of "
    "reasoning, then the answer on its own line, without labels."
)
EFFORT_STYLE = {
    "gguf": {"low": FEWSHOT_STYLE, "medium": FEWSHOT_STYLE},
}


# Backends with no distinct policy fall back to the gguf table: a missing
# family means an unknown profile, and budgeting like Qwen is the safe default.
def effort_policy(family):
    return EFFORT_POLICY.get(family, EFFORT_POLICY["gguf"])
EFFORT_ALIASES = {"minimal": "low"}
BUDGET_KEYS = ("reasoning_budget_tokens", "thinking_budget_tokens")

# Injected right before the forced end-of-thinking tag when the budget runs
# out. Without it llama.cpp forces a bare </think> mid-sentence; Bonsai then
# frequently emits EOS and the turn ends with no answer at all.
BUDGET_MESSAGE = "\n\nOkay, I have thought enough. Time to write the final answer."

# Leave the model room to answer after the forced transition. 1,536 covers
# the measured answers (6-1,100 tokens) while still letting a 4k request think
# for 2,560 tokens. A tiny max_tokens halves instead of starving the answer.
ANSWER_RESERVE = 1536


def effective_budget(configured, decoded):
    """Cap thinking so the answer still fits the client's max_tokens.

    high/xhigh are configured unbounded; with a 4k max_tokens the model used
    all 4k thinking and returned an empty answer. The cap is dynamic per
    request, not a fixed number.
    """
    limit = decoded.get("max_tokens")
    if not isinstance(limit, int) or limit <= 0:
        limit = decoded.get("max_completion_tokens")
    if not isinstance(limit, int) or limit <= 0:
        return configured
    room = limit - ANSWER_RESERVE
    if room < 256:
        room = max(128, limit // 2)
    return room if configured is None else min(configured, room)


def inject_style(decoded, path, style):
    """Prepend the style example to the client's system prompt, not replace it."""
    if path.startswith("/v1/messages"):
        system = decoded.get("system")
        if isinstance(system, list):
            decoded["system"] = [{"type": "text", "text": style}] + system
        elif isinstance(system, str) and system:
            decoded["system"] = style + "\n\n" + system
        else:
            decoded["system"] = style
        return
    messages = decoded.get("messages")
    if not isinstance(messages, list):
        return
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [{"type": "text", "text": style}] + content
            elif isinstance(content, str) and content:
                msg["content"] = style + "\n\n" + content
            else:
                msg["content"] = style
            return
    messages.insert(0, {"role": "system", "content": style})


def parse_model(model):
    """Split '<base>[-effort]' into (profile, effort|None).

    Effort is the trailing '-<effort>' segment (xhigh matched before high).
    The '-max' tier marker is not an effort: 'qwen3.8-gguf-max' has no suffix,
    'qwen3.8-gguf-max-medium' is tier max + effort medium.
    """
    effort = None
    base = model if isinstance(model, str) else ""
    for suffix in ("-xhigh", "-high", "-medium", "-low", "-none",
                   "-minimal"):
        if base.endswith(suffix):
            effort = suffix[1:]
            base = base[: -len(suffix)]
            break
    if "bonsai" in base:
        # Ternary Bonsai 2 27B tiers. Match the tier word, not the full id:
        # the client catalog uses bonsai-2-27b-<tier>.
        for tier in ("trio256", "trio200", "duo256", "hexa128", "penta128",
                     "octo64", "quad128", "duo128", "vision", "max"):
            if tier in base:
                return f"bonsai-{tier}", effort
        return "bonsai-max", effort
    if "gguf-vision" in base:
        return "gguf-vision", effort
    if "gguf-speed" in base:
        return "gguf-speed", effort
    if "gguf-duo128" in base:
        return "gguf-duo128", effort
    if "gguf-duo" in base:
        return "gguf-duo", effort
    if "gguf-quad" in base:
        return "gguf-quad", effort
    if "gguf-single192" in base:
        return "gguf-single192", effort
    if "gguf-max" in base:
        return "gguf-max", effort
    if "gguf-range" in base or "gguf" in base:
        return "gguf-range", effort
    # Unknown names land on the general-purpose Qwen tier: the server accepts
    # arbitrary model strings, so routing them anywhere else would silently
    # pick the wrong weights.
    return "gguf-range", effort


def has_media(decoded):
    """True when any message carries image/video content parts.

    Covers OpenAI (image_url, video_url) and Anthropic (image, document
    with image media) shapes. A media request needs the vision backend no
    matter which model name the client sent.
    """
    try:
        for msg in decoded.get("messages", []) or []:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    t = part.get("type")
                    if t in ("image_url", "input_image", "image",
                             "video_url", "input_video"):
                        return True
                    source = part.get("source") or {}
                    if source.get("media_type", "").startswith("image/"):
                        return True
    except Exception:
        pass
    return False


def message_digest(msg):
    """Short stable hash of one chat message, tool calls included."""
    raw = json.dumps(msg, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def prompt_prefix_key(decoded, profile=None):
    """Hash the request prefix, omitting only the final chat message.

    The final message normally changes on every turn; the preceding messages,
    tools, and system prompt are what llama.cpp can reuse from its KV cache.
    Keep this conservative: requests without a message prefix are always
    considered cold.
    """
    messages = decoded.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    prefix = dict(decoded)
    if profile:
        # A tier switch/restart destroys the backend KV cache even when the
        # chat prefix itself is identical.
        prefix["_router_backend_profile"] = profile
    prefix["messages"] = messages[:-1]
    raw = json.dumps(prefix, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()


def prompt_request_key(decoded, profile=None):
    """Hash the complete admitted request for the next turn's prefix match."""
    messages = decoded.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    request = dict(decoded)
    if profile:
        request["_router_backend_profile"] = profile
    raw = json.dumps(request, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()


def prompt_is_hot(decoded, profile=None):
    """Return whether this request's reusable prefix was admitted recently."""
    key = prompt_prefix_key(decoded, profile)
    if key is None:
        return False
    now = time.monotonic()
    with prefix_lock:
        while recent_prefixes:
            _, seen = next(iter(recent_prefixes.items()))
            if now - seen <= PREFILL_PREFIX_TTL:
                break
            recent_prefixes.popitem(last=False)
        seen = recent_prefixes.get(key)
        if seen is None:
            return False
        recent_prefixes.move_to_end(key)
        recent_prefixes[key] = now
        return True


def remember_prompt_prefix(decoded, profile=None):
    # Store the complete admitted request: on the next turn, the client sends
    # that conversation as the prefix and appends one new final message.
    key = prompt_request_key(decoded, profile)
    if key is None:
        return
    now = time.monotonic()
    with prefix_lock:
        recent_prefixes[key] = now
        recent_prefixes.move_to_end(key)
        while len(recent_prefixes) > PREFILL_PREFIX_MAX:
            recent_prefixes.popitem(last=False)


def next_token_info(slot):
    """(n_decoded, has_next_token) for one /slots entry, defensively."""
    nt = slot.get("next_token")
    if not isinstance(nt, list) or not nt or not isinstance(nt[0], dict):
        return 0, False
    return int(nt[0].get("n_decoded") or 0), bool(nt[0].get("has_next_token"))


def slot_phase(slot):
    """Classify one /slots entry, or None when it is unusable.

    llama.cpp does not expose the whole prompt length: `n_prompt_tokens` is
    what already sits in the KV cache for this task plus the batch queued for
    the next decode step. So `cache + processed` never reaches it and the only
    reliable prefill signal is "processing, nothing sampled yet":

      prefilling -- is_processing, no next token, 0 decoded, and enough of
                    this task's own prompt in flight to matter;
      generating -- is_processing and a token was sampled for it.

    `in_flight` = n_prompt_tokens - cached is the part of this task that is
    not reusable prefix (processed so far plus the queued batch).
    """
    if not isinstance(slot, dict):
        return None
    try:
        sid = int(slot.get("id", -1))
        proc = bool(slot.get("is_processing"))
        decoded, has_next = next_token_info(slot)
        n_prompt = int(slot.get("n_prompt_tokens") or 0)
        cached = int(slot.get("n_prompt_tokens_cache") or 0)
        processed = int(slot.get("n_prompt_tokens_processed") or 0)
    except (TypeError, ValueError):
        return None
    prompt_len = max(0, n_prompt - decoded)
    in_flight = max(0, n_prompt - cached)
    queued = max(0, prompt_len - cached - processed)
    prefilling = (proc and decoded == 0 and not has_next
                  and in_flight >= PREFILL_GATE_MIN)
    generating = proc and (has_next or decoded > 0)
    return {"id": sid, "proc": proc, "decoded": decoded,
            "n_prompt": n_prompt, "prompt_len": prompt_len,
            "cached": cached, "processed": processed,
            "in_flight": in_flight, "queued": queued,
            "prefilling": prefilling, "generating": generating}


def backend_slot_state():
    """Return the backend's slot phases, or None when /slots is unavailable."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=3)
        conn.request("GET", "/slots")
        resp = conn.getresponse()
        raw = resp.read(4 * 1024 * 1024)
        conn.close()
        if resp.status != 200:
            return None
        slots = json.loads(raw)
    except Exception:
        return None
    if not isinstance(slots, list):
        return None
    summary = []
    for slot in slots:
        phase = slot_phase(slot)
        if phase is not None:
            summary.append(phase)
    return {"prefilling": any(s["prefilling"] for s in summary),
            "generating": any(s["generating"] for s in summary),
            "slots": summary}


def gate_decision(prefilling, generating, cold_large):
    """Why this admission should wait, or None to send it now.

    Phase A: an arrival is held while another slot is still prefilling. Phase B
    (opt-in): a cold large prompt is held while any slot is decoding, which is
    the case that drops a co-tenant to ~1 t/s.
    """
    if prefilling:
        return "prefill"
    if cold_large and generating:
        return "cold-vs-decode"
    return None


def admit_prefill(decoded, body, profile=None):
    """Hold a request while another slot is prefilling (and optionally while a
    slot is decoding a cold large prompt).

    Best-effort by design: it must never fail a request. Any problem reading
    the backend degrades to "send it now".
    """
    if not PREFILL_GATE or not isinstance(decoded, dict):
        return
    try:
        est = len(body or b"") / 3.5
        hot = prompt_is_hot(decoded, profile)
        cold_large = (PREFILL_GATE_COLD and not hot
                      and est >= PREFILL_GATE_COLD_MIN)
        phase = backend_slot_state()
        if phase is None:
            if PREFILL_GATE_DIAG and PREFILL_GATE_DIAG_VERBOSE:
                log(f"prefill gate diag slots=unavailable est={int(est)}")
            return
        reason = gate_decision(phase["prefilling"], phase["generating"],
                               cold_large)
        if PREFILL_GATE_DIAG and (reason or PREFILL_GATE_DIAG_VERBOSE):
            slots = "; ".join(
                f"s{sl['id']}:p={int(sl['proc'])},d={sl['decoded']},"
                f"np={sl['n_prompt']},pl={sl['prompt_len']},"
                f"c={sl['cached']},pp={sl['processed']},"
                f"inflight={sl['in_flight']},queued={sl['queued']}"
                for sl in phase["slots"])
            log(f"prefill gate diag est={int(est)} hot={hot} "
                f"cold={cold_large} prefilling={phase['prefilling']} "
                f"generating={phase['generating']} "
                f"would_block={reason is not None} reason={reason or 'pass'} "
                f"slots=[{slots}]")
        if reason and PREFILL_GATE_ENFORCE:
            started = time.monotonic()
            waited = False
            while reason:
                if time.monotonic() - started >= PREFILL_GATE_TIMEOUT:
                    log(f"prefill gate timeout reason={reason} "
                        f"cold={cold_large} est={int(est)}")
                    break
                if not waited:
                    log(f"prefill gate wait reason={reason} est={int(est)}")
                    waited = True
                time.sleep(PREFILL_GATE_POLL)
                phase = backend_slot_state()
                if phase is None:
                    break
                reason = gate_decision(phase["prefilling"],
                                       phase["generating"], cold_large)
            if waited:
                log("prefill gate admit waited="
                    f"{time.monotonic() - started:.1f}s")
    except Exception as e:  # a gate bug must never fail a request
        log(f"prefill gate skipped: {e!r}")
    finally:
        try:
            remember_prompt_prefix(decoded, profile)
        except Exception:
            pass



def debug_prompt(decoded, body):
    """Prefix-cache forensics (opt-in via env).

    llama.cpp only reuses a prompt up to the first token that differs from the
    previous one, so a single message that changes early in the list costs a
    full re-prefill of the whole history (tens of seconds at 100K+ tokens).
    One hash per message makes that first difference obvious in the log;
    MODEL_ROUTER_DUMP=<dir> keeps whole bodies for an exact offline diff.
    """
    dump = os.environ.get("MODEL_ROUTER_DUMP")
    if not os.environ.get("MODEL_ROUTER_DIGEST") and not dump:
        return
    msgs = decoded.get("messages") or []
    head = " ".join(f"{i}={message_digest(m)}" for i, m in enumerate(msgs))
    log(f"prompt n_msgs={len(msgs)} "
        f"n_tools={len(decoded.get('tools') or [])} "
        f"bytes={len(body or b'')} digests {head}")
    if dump:
        try:
            os.makedirs(dump, exist_ok=True)
            path = os.path.join(dump, f"{time.time_ns() // 1000000}.json")
            with open(path, "wb") as f:
                f.write(body if isinstance(body, bytes) else body.encode())
        except Exception as e:  # diagnostics must never break a request
            log(f"prompt dump failed: {e}")


def route_request(decoded):
    """Return (profile, suffix_effort, want_mmproj, notes) for a request.

    Media on any GGUF/Bonsai tier wants mmproj on that SAME tier (lazy
    upgrade), never a tier switch: every backend here can carry the encoder.
    """
    profile, suffix_effort = parse_model(decoded.get("model"))
    notes = []
    want_mmproj = profile in ("gguf-vision", "bonsai-vision")
    if has_media(decoded) and profile in LLAMA_ALL:
        # Same tier, lazy encoder: reboot it with mmproj instead of switching
        # tiers (prefix cache is lost either way; tier stays).
        want_mmproj = True
        notes.append("media wants mmproj")
    return profile, suffix_effort, want_mmproj, notes


def resolve_effort(decoded, suffix_effort):
    """Return (canonical tier, origin) from body, kwargs, or model suffix."""
    effort = decoded.get("reasoning_effort")
    origin = "client"
    if not isinstance(effort, str) or not effort.strip():
        ctk = decoded.get("chat_template_kwargs")
        effort = ctk.get("reasoning_effort") if isinstance(ctk, dict) else None
        origin = "client-kwargs"
    if not isinstance(effort, str) or not effort.strip():
        effort = suffix_effort
        origin = "suffix"
    if not isinstance(effort, str) or not effort.strip():
        return None, None
    effort = effort.strip().lower()
    return EFFORT_ALIASES.get(effort, effort), origin


def client_thinking_mode(decoded, path):
    """'enabled'/'disabled' when the client set thinking explicitly."""
    if path.startswith("/v1/messages"):
        thinking = decoded.get("thinking")
        if isinstance(thinking, dict):
            t = thinking.get("type")
            if t in ("enabled", "disabled"):
                return t
    ctk = decoded.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        value = ctk.get("enable_thinking")
        if value is False or value == "false":
            return "disabled"
        if value is True or value == "true":
            return "enabled"
    return None


def has_explicit_budget(decoded, path):
    if any(k in decoded for k in BUDGET_KEYS):
        return True
    if path.startswith("/v1/messages"):
        thinking = decoded.get("thinking")
        if (isinstance(thinking, dict)
                and thinking.get("type") == "enabled"
                and "budget_tokens" in thinking):
            return True
        extra = decoded.get("extra_params")
        if isinstance(extra, dict) and any(k in extra for k in BUDGET_KEYS):
            return True
    return False


def apply_effort_policy(decoded, path, effort, family, meta, notes):
    """Rewrite one request to the tier's template, style and dynamic budget.

    Explicit client thinking wins: a disabled thinking block is never
    re-enabled, and an explicit budget is never overwritten. `family` selects
    the policy; an unknown family falls back to the gguf table.
    """
    template, configured = effort_policy(family)[effort]
    meta["template"] = template
    mode = client_thinking_mode(decoded, path)
    if mode == "disabled":
        notes.append("thinking disabled by client")
        return False
    if effort == "none":
        if mode == "enabled":
            notes.append("thinking enabled by client")
            return False
        if path.startswith("/v1/messages"):
            ctk = decoded.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = {}
                decoded["chat_template_kwargs"] = ctk
            ctk["enable_thinking"] = False
            ctk.pop("reasoning_effort", None)
            decoded.pop("reasoning_effort", None)
        else:
            decoded["reasoning_effort"] = "none"
        notes.append("effort none")
        return True
    style = EFFORT_STYLE.get(family, {}).get(effort)
    if style:
        inject_style(decoded, path, style)
        meta["style"] = True
        notes.append("style injected")
    budget = effective_budget(configured, decoded)
    if path.startswith("/v1/messages"):
        ctk = decoded.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
            decoded["chat_template_kwargs"] = ctk
        ctk["enable_thinking"] = True
        ctk["reasoning_effort"] = template
        decoded.pop("reasoning_effort", None)
        notes.append(f"effort {effort}->{template}")
        if budget is not None:
            if has_explicit_budget(decoded, path):
                meta["budget"] = "client"
                notes.append("budget kept from client")
            else:
                thinking = decoded.get("thinking")
                if (isinstance(thinking, dict)
                        and thinking.get("type") == "enabled"):
                    thinking["budget_tokens"] = budget
                else:
                    decoded["thinking"] = {"type": "enabled",
                                           "budget_tokens": budget}
                meta["budget"] = budget
                meta["injected_budget"] = True
                notes.append(f"budget {budget}")
        return True
    decoded["reasoning_effort"] = template
    notes.append(f"effort {effort}->{template}")
    capped = has_explicit_budget(decoded, path)
    if capped:
        meta["budget"] = "client"
        notes.append("budget kept from client")
    elif budget is not None:
        decoded["reasoning_budget_tokens"] = budget
        meta["budget"] = budget
        meta["injected_budget"] = True
        capped = True
        notes.append(f"budget {budget}")
    # A bare forced </think> lands mid-sentence and the model often answers
    # with EOS instead of an answer (Bonsai: ~50% of capped turns in long
    # agentic contexts). Give it a transition first, whoever set the budget.
    # An explicit client message still wins.
    if capped:
        decoded.setdefault("reasoning_budget_message", BUDGET_MESSAGE)
    return True


def normalize_chat_body(path, body, route=None, final_profile=None):
    """Return (profile, new_body, log_note, want_mmproj, meta).

    - Routes the profile from the model name (route_request).
    - Maps the client tier (body, chat_template_kwargs, or -effort suffix)
      through effort_policy(family): template hint + reasoning budget.
    - Budgets only reach llama.cpp backends (final_profile in LLAMA_ALL).
    Explicit client values always win.
    """
    decoded = json.loads(body or b"{}")
    if route is None:
        route = route_request(decoded)
    requested, suffix_effort, want_mmproj, notes = route
    notes = list(notes)
    profile = requested if final_profile is None else final_profile
    modified = False
    meta = {"effort": None, "origin": None, "template": None,
            "budget": None, "injected_budget": False}
    effort, origin = resolve_effort(decoded, suffix_effort)
    meta["effort"] = effort
    meta["origin"] = origin
    if effort and effort not in EFFORT_POLICY["gguf"]:
        notes.append(f"ignored unsupported effort {effort}")
        if path.startswith("/v1/chat/completions"):
            decoded.pop("reasoning_effort", None)
        else:
            ctk = decoded.get("chat_template_kwargs")
            if isinstance(ctk, dict):
                ctk.pop("reasoning_effort", None)
        modified = True
        effort = None
        meta["effort"] = None
    if effort:
        changed = apply_effort_policy(
            decoded, path, effort, family(profile), meta, notes)
        modified = modified or changed
    # Skip re-serialization when untouched: json round-trips of multi-MB
    # histories are pure router latency (and transient 2x memory).
    new_body = json.dumps(decoded).encode() if modified else (body or b"{}")
    note = (" (" + ", ".join(notes) + ")") if notes else ""
    return profile, new_body, note, want_mmproj, meta


def same_backend(cur_name, cur_mmproj, profile, want_mmproj):
    """True when the running backend already satisfies the request.

    `state["profile"]` only ever holds the profile *name*; the encoder
    (mmproj) state lives separately in `state["profile_mmproj"]`.

    A resident encoder is sticky: it serves text requests too. Reboot only to
    *add* mmproj when media arrives. Toggling it off for text would restart a
    multi-slot 20+ GiB backend (full re-prefill) just to reclaim 0.6 GiB, and
    a mixed image/text session would ping-pong on every turn.
    """
    return cur_name == profile and (bool(cur_mmproj) or not bool(want_mmproj))


def ensure_backend(profile, want_mmproj=False):
    """Must hold `lock`. Returns True when the wanted profile serves."""
    name = state["profile"]
    cur_mmproj = state["profile_mmproj"]
    # Same profile and (encoder resident or not needed) → reuse. Only a media
    # request against a text-only backend reboots; the encoder is never
    # removed just to reclaim memory.
    if same_backend(name, cur_mmproj, profile, want_mmproj) and \
            (backend_container() or backend_healthy()):
        return True
    if backend_container() or state["profile"]:
        stop_backend()
    # Wait for the port to actually free (container teardown races).
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            c = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=3)
            c.request("GET", "/health")
            c.getresponse()
            time.sleep(2)
        except Exception:
            break
    return start_backend(profile, want_mmproj)


def idle_watch():
    while True:
        time.sleep(30)
        with lock:
            if (state["profile"] and state["active"] == 0
                    and time.time() - state["last_used"] > IDLE_SEC):
                log(f"idle {IDLE_SEC}s: unloading {state['profile']}")
                stop_backend()


STATIC_MODELS = {
    "object": "list",
    "data": [
        # Qwen3.8 27B (llama.cpp GGUF). Context is per slot.
        {"id": "qwen3.8-gguf-duo128", "object": "model",
         "owned_by": "llama.cpp", "name": "qwen3.8-gguf-duo128",
         "context_window": 131072, "max_model_len": 131072,
         "modalities": {"vision": False}},
        {"id": "qwen3.8-gguf-quad", "object": "model",
         "owned_by": "llama.cpp", "name": "qwen3.8-gguf-quad",
         "context_window": 65536, "max_model_len": 65536,
         "modalities": {"vision": False}},
        {"id": "qwen3.8-gguf-single192", "object": "model",
         "owned_by": "llama.cpp", "name": "qwen3.8-gguf-single192",
         "context_window": 196608, "max_model_len": 196608,
         "modalities": {"vision": False}},
        {"id": "qwen3.8-gguf-max", "object": "model", "owned_by": "llama.cpp",
         "name": "qwen3.8-gguf-max", "context_window": 262144,
         "max_model_len": 262144, "modalities": {"vision": False}},
        # Ternary Bonsai 2 27B (PrismML fork runtime). Context is per slot.
        {"id": "bonsai-2-27b-max", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-max", "context_window": 262144,
         "max_model_len": 262144, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-duo256", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-duo256", "context_window": 262144,
         "max_model_len": 262144, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-trio256", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-trio256", "context_window": 262144,
         "max_model_len": 262144, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-trio200", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-trio200", "context_window": 204800,
         "max_model_len": 204800, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-duo128", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-duo128", "context_window": 131072,
         "max_model_len": 131072, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-quad128", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-quad128", "context_window": 131072,
         "max_model_len": 131072, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-penta128", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-penta128", "context_window": 131072,
         "max_model_len": 131072, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-hexa128", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-hexa128", "context_window": 131072,
         "max_model_len": 131072, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-octo64", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-octo64", "context_window": 65536,
         "max_model_len": 65536, "modalities": {"vision": False}},
        {"id": "bonsai-2-27b-vision", "object": "model", "owned_by": "prism",
         "name": "bonsai-2-27b-vision", "context_window": 65536,
         "max_model_len": 65536, "modalities": {"vision": True}},
    ],
}

HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "upgrade"}

RETRYABLE_MARKERS = (b"expired", b"admission", b"overloaded", b"busy",
                      b"try again", b"queue", b"unavailable")
MAX_RETRIES = 3


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "model-router"

    def _send(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        return self.rfile.read(n) if n > 0 else b""

    def do_GET(self):
        # The slot API can save/restore files on the backend and is never part
        # of the public surface: the router talks to it directly instead.
        if self.path.startswith("/slots"):
            return self._send(403, json.dumps(
                {"error": "slot API is not proxied"}))
        if self.path == "/health":
            # Lock-free on purpose: a wedged boot must never take down the
            # status endpoint (single-key reads are GIL-atomic; worst case
            # the snapshot is a moment stale).
            return self._send(200, json.dumps({"status": "router-ok",
                                              "backend": state["profile"],
                                              "busy": state["busy"]}))
        if self.path == "/v1/models":
            up = state["profile"] and (backend_container() or backend_healthy())
            if up:
                return self._relay(None)
            # Idle: advertise both variants so pickers work cold.
            return self._send(200, json.dumps(STATIC_MODELS))
        return self._relay(None)

    def do_POST(self):
        if self.path.startswith("/slots"):
            return self._send(403, json.dumps(
                {"error": "slot API is not proxied"}))
        body = self._read_body()
        profile = None
        if self.path.startswith("/v1/chat/completions") or \
                self.path.startswith("/v1/messages"):
            try:
                decoded = json.loads(body or b"{}")
                route = route_request(decoded)
                requested, _suffix, want_mmproj, route_notes = route
                # Reuse the running backend when it can serve: a switch wipes
                # the prefix cache (full re-prefill). Estimate prompt tokens
                # conservatively from body bytes; overestimate switches more.
                need_vision = has_media(decoded)
                est = len(body) / 3.5 if body else 0
                reused = effective_profile(requested, state["profile"], est,
                                           need_vision)
                profile = reused
                note = (" (" + ", ".join(route_notes) + ")") \
                    if route_notes else ""
                if reused != requested:
                    note += f" (reuse {reused}, fits)"
                _, body, norm_note, norm_mmproj, meta = normalize_chat_body(
                    self.path, body, route=route, final_profile=profile)
                want_mmproj = want_mmproj or norm_mmproj
                if need_vision and profile in LLAMA_ALL:
                    want_mmproj = True
                note += norm_note
                decoded = json.loads(body)
                log(f"chat model={decoded.get('model')} "
                    f"reasoning_effort={decoded.get('reasoning_effort', '<absent>')} "
                    f"effort={meta.get('effort') or '<absent>'} "
                    f"template={meta.get('template') or '<absent>'} "
                    f"budget={meta.get('budget') if meta.get('budget') is not None else '<absent>'} "
                    f"origin={meta.get('origin') or '<absent>'}{note} "
                    f"max_tokens={decoded.get('max_tokens', '<absent>')} "
                    f"stream={decoded.get('stream', False)} -> {profile} "
                    f"mmproj={want_mmproj}")
                debug_prompt(decoded, body)
            except Exception:
                profile = None
                want_mmproj = False
            return self._relay(body, profile, want_mmproj)

    def _relay(self, body, profile=None, want_mmproj=False):
        with lock:
            if profile and profile in PROFILES:
                state["busy"] = True
                try:
                    if not ensure_backend(profile, want_mmproj):
                        state["busy"] = False
                        return self._send(502, json.dumps(
                            {"error": f"backend profile {profile} failed to start"}))
                finally:
                    state["busy"] = False
            state["active"] += 1
        try:
            if (profile and body and self.command == "POST" and
                    (self.path.startswith("/v1/chat/completions") or
                     self.path.startswith("/v1/messages"))):
                try:
                    admit_prefill(json.loads(body), body, profile)
                except Exception as e:  # never fail a turn on gate parsing
                    log(f"prefill gate skipped: {e!r}")
            fwd = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_HEADERS}
            # Body may have been rewritten (effort injection): length must match.
            if body is not None:
                fwd["Content-Length"] = str(len(body))
            else:
                fwd.pop("Content-Length", None)
                fwd.pop("content-length", None)
            # Single-slot backends serialize bursts; a queued request may be
            # refused (503/admission expiry). Generations are side-effect
            # free, so retry those transparently instead of failing the turn.
            retryable = (self.command == "POST" and
                         (self.path.startswith("/v1/chat/completions") or
                          self.path.startswith("/v1/messages")))
            resp = None
            for attempt in range(MAX_RETRIES + 1):
                conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT,
                                                  timeout=3600)
                conn.request(self.command, self.path, body=body or None,
                             headers=fwd)
                resp = conn.getresponse()
                if resp.status == 200:
                    break
                err = resp.read(65536) or b""
                conn.close()
                if (retryable and attempt < MAX_RETRIES
                        and (resp.status in (429, 503) or any(
                            m in err.lower() for m in RETRYABLE_MARKERS))):
                    wait = 5 * (attempt + 1)
                    log(f"backend busy ({resp.status}), retry {attempt + 1} "
                        f"in {wait}s: {err[:120]!r}")
                    time.sleep(wait)
                    continue
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.getheader(
                    "Content-Type", "application/octet-stream"))
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                try:
                    self.wfile.write(err)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                with lock:
                    state["last_used"] = time.time()
                return
            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type", "application/octet-stream")
            self.send_header("Content-Type", ctype)
            length = resp.getheader("Content-Length")
            is_sse = "text/event-stream" in ctype
            if not is_sse and length is not None:
                self.send_header("Content-Length", length)
            self.end_headers()
            if is_sse:
                # Forward event-by-event: block reads batch tokens into
                # bursts and the tail hangs without EOF propagation.
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                self.close_connection = True
            else:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
            with lock:
                state["last_used"] = time.time()
        except Exception as e:
            try:
                self._send(502, json.dumps({"error": f"backend error: {e}"}))
            except Exception:
                pass
        finally:
            with lock:
                state["active"] -= 1

    def log_message(self, *a):
        pass


def infer_profile(args):
    """Best-effort profile name from a backend container's server args.

    Returns None when the args match no known launcher. Used to *adopt* an
    already-running backend on router start instead of rebooting it.
    """
    if not args:
        return None
    toks = args.split()
    # llama.cpp: the routing name rides on --alias (its entrypoint, not an
    # arg, is what runs llama-server).
    for i, t in enumerate(toks):
        if t == "--alias" and i + 1 < len(toks):
            return parse_model(toks[i + 1])[0]
        if t.startswith("--alias="):
            return parse_model(t.split("=", 1)[1])[0]
    # llama.cpp without an alias: fall back to shape heuristics.
    if "Ternary-Bonsai" in args:
        if "--parallel 8" in args:
            return "bonsai-octo64"
        if "--parallel 4" in args:
            return "bonsai-quad128"
        if "--parallel 6" in args:
            return "bonsai-hexa128"
        if "--parallel 5" in args:
            return "bonsai-penta128"
        if "--parallel 3" in args:
            return ("bonsai-trio200"
                    if "-c 614400" in args or "--ctx-size 614400" in args
                    else "bonsai-trio256")
        if "--parallel 2" in args:
            return ("bonsai-duo256"
                    if "-c 524288" in args or "--ctx-size 524288" in args
                    else "bonsai-duo128")
        if "mmproj" in args:
            return "bonsai-vision"
        return "bonsai-max"
    if "--parallel 8" in args:
        return "bonsai-octo64" if "Ternary-Bonsai" in args else "gguf-quad"
    if "--parallel 4" in args:
        return "gguf-quad"
    if "--parallel 2" in args:
        return "gguf-duo128" if "262144" in args else "gguf-duo"
    if "-c 262144" in args or "--ctx-size 262144" in args:
        return "gguf-max"
    if "mtp.gguf" in args:
        return "gguf-speed"
    if "mmproj" in args:
        return "gguf-vision"
    if "-m " in args:
        return "gguf-range"
    return None


if __name__ == "__main__":
    # Adopt an already-running backend instead of rebooting it: infer the
    # profile from the container's server args (llama --alias / shape).
    cid = backend_container()
    if cid:
        rc, args = sh("docker", "inspect", "-f", "{{join .Args \" \"}}", cid)
        if rc == 0:
            state["profile"] = infer_profile(args)
            if state["profile"]:
                state["profile_mmproj"] = "mmproj" in args
            log(f"adopted running backend {cid} as {state['profile']} "
                f"mmproj={state['profile_mmproj']}")
    threading.Thread(target=idle_watch, daemon=True).start()
    log(f"router :{ROUTER_PORT} -> backend :{BACKEND_PORT} (idle {IDLE_SEC}s)")
    ThreadingHTTPServer(("127.0.0.1", ROUTER_PORT), Handler).serve_forever()
