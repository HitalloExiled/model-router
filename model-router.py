#!/usr/bin/env python3
"""NInfer model router: Ollama-style load/unload across serve.sh profiles.

Listens on ROUTER_PORT (default 18089) and forwards to a single ninfer-serve
backend on BACKEND_PORT (default 8080). The requested OpenAI model name picks
the backend profile:

  qwen3.8-27b-max   -> serve.sh --profile max      (192K, rk4v4-e8)
  *256k*             -> serve.sh --profile ultra    (256K, rk2v4-e8)
  *64k*             -> serve.sh --profile c1        (64K, int8, widest KV)
  qwen3.8-27b*      -> serve.sh --profile balanced (128K, rk8v4)

Any other model name reuses whatever backend is running (the server accepts
arbitrary model strings). Switching profiles stops the running container and
boots the wanted one (~60s cold start); after IDLE_SEC seconds without
requests the backend is stopped to free VRAM.

Env:
  MODEL_ROUTER_PORT / NINFER_BACKEND_PORT / MODEL_ROUTER_IDLE_SEC (default 900)
  NINFER_MODEL_DIR / NINFER_DOCKER_IMAGE (forwarded to serve.sh)
  MODEL_ROUTER_DIGEST=1 / MODEL_ROUTER_DUMP=<dir> (prefix-cache forensics)
"""
import hashlib
import http.client
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))


def stack_dir(name):
    """Locate a stack checkout: inside the router repo (git submodule) first,
    then as a sibling checkout (~/Projetos/My/<name>)."""
    inside = os.path.join(ROUTER_DIR, name)
    return inside if os.path.isdir(inside) else os.path.join(
        os.path.dirname(ROUTER_DIR), name)


SERVE = os.environ.get(
    "NINFER_SERVE_SH", os.path.join(stack_dir("ninfer-stack"), "serve.sh"))
NINFER_MODELS_DEFAULT = os.path.join(stack_dir("ninfer-stack"), "models")
MODEL_FILE = os.path.join(
    os.environ.get("NINFER_MODEL_DIR", NINFER_MODELS_DEFAULT),
    "qwen3_8_27b.ninfer",
)
VISION_FILE = os.path.join(
    os.environ.get("NINFER_MODEL_DIR", NINFER_MODELS_DEFAULT),
    "qwen3_6_35b_a3b.ninfer",
)
# Backend artifact per profile; everything text runs the Qwen3.8 weights.
PROFILE_MODELS = {"vision": VISION_FILE}
IMAGE = os.environ.get("NINFER_DOCKER_IMAGE", "ninfer-4090:sm89")
# llama-server stack (GGUF pilot): second image + artifacts. Flags live in
# scripts/serve-llama.sh; the router only selects profile -> argv here.
LLAMA_IMAGE = os.environ.get(
    "NINFER_LLAMA_IMAGE",
    "ghcr.io/ggml-org/llama.cpp:server-cuda-b11854")
GGUF_DIR = os.environ.get(
    "NINFER_GGUF_DIR", os.path.join(stack_dir("gguf-stack"), "models"))
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
BONSAI_DIR = os.environ.get(
    "NINFER_BONSAI_DIR", os.path.join(stack_dir("bonsai-stack"), "models"))
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
LOG_DIR = os.path.join(os.path.expanduser("~"), ".local", "share", "ninfer")
BOOT_LOG = os.path.join(LOG_DIR, "ninfer-backend-boot.log")
ROUTER_PORT = int(os.environ.get("MODEL_ROUTER_PORT", "18089"))
BACKEND_PORT = int(os.environ.get("NINFER_BACKEND_PORT", "8080"))
IDLE_SEC = int(os.environ.get("MODEL_ROUTER_IDLE_SEC", "900"))
BOOT_TIMEOUT = 420

PROFILES = {"balanced", "max", "c1", "ultra", "vision",
            "gguf-speed", "gguf-range", "gguf-max", "gguf-duo", "gguf-duo128",
            "gguf-quad", "gguf-vision", "gguf-single192",
            "bonsai-max", "bonsai-duo256", "bonsai-trio256", "bonsai-trio200",
            "bonsai-duo128",
            "bonsai-quad128", "bonsai-penta128", "bonsai-hexa128",
            "bonsai-octo64", "bonsai-vision"}
# Per-slot window (a client sees one slot, not the whole pool).
PROFILE_WINDOWS = {"c1": 65536, "balanced": 131072, "max": 196608,
                   "ultra": 262144, "vision": 16384,
                   "gguf-speed": 65536, "gguf-range": 98304,
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
PROFILE_SLOTS = {"c1": 1, "balanced": 1, "max": 1, "ultra": 1, "vision": 1,
                 "gguf-speed": 1, "gguf-range": 1, "gguf-max": 1,
                 "gguf-duo": 2, "gguf-duo128": 2, "gguf-quad": 4,
                 "gguf-vision": 1, "gguf-single192": 1,
                 "bonsai-max": 1, "bonsai-duo256": 2, "bonsai-trio256": 3,
                 "bonsai-trio200": 3,
                 "bonsai-duo128": 2, "bonsai-quad128": 4,
                 "bonsai-penta128": 5, "bonsai-hexa128": 6,
                 "bonsai-octo64": 8, "bonsai-vision": 1}
# Every GGUF backend CAN carry mmproj (lazy); only ninfer text tiers need a
# switch for media (to 35B vision).
VISION_CAPABLE = {"vision", "gguf-speed", "gguf-range", "gguf-max",
                  "gguf-duo", "gguf-duo128", "gguf-quad", "gguf-vision",
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

# Backend families. Reuse never crosses families: answering a qwen3.8-gguf-*
# request from a resident Bonsai backend (or the reverse) returns a different
# model than the client asked for, which is a correctness bug, not a cache
# win. Inside a family every tier serves the same weights, so the wider tier
# can absorb the request and keep the prefix cache.
NINFER_ALL = {"balanced", "max", "c1", "ultra", "vision"}


def family(profile):
    if profile in BONSAI_ALL:
        return "bonsai"
    if profile in GGUF_ALL:
        return "gguf"
    if profile in NINFER_ALL:
        return "ninfer"
    return None


def effective_profile(wanted, current, est_tokens, need_vision):
    """Prefer the running backend over the requested tier.

    A tier switch wipes the server prefix cache (full_reset), forcing a full
    re-prefill (~60s at 100K tokens). Switch only when the running backend
    cannot serve: different model family, media needs vision, the estimated
    prompt exceeds 90% of its window, or the request asks for more parallel
    slots than the resident backend has. That last clause is what stops a
    1-slot backend from silently serializing requests meant for a 2/4/8-slot
    tier.
    """
    if current in PROFILE_WINDOWS:
        # Every GGUF/Bonsai backend carries mmproj: media never forces a
        # switch off them. Only ninfer text tiers must yield to 35B vision.
        if need_vision and current not in VISION_CAPABLE:
            return wanted
        if family(current) != family(wanted):
            return wanted
        enough_slots = (not RESPECT_SLOTS
                        or PROFILE_SLOTS.get(current, 1)
                        >= PROFILE_SLOTS.get(wanted, 1))
        if enough_slots and est_tokens < 0.9 * PROFILE_WINDOWS[current]:
            return current
    return wanted


lock = threading.Lock()
state = {"profile": None, "profile_mmproj": False, "busy": False,
         "last_used": 0.0, "active": 0,
         "current_model": None, "current_profile": None}


def log(*a):
    print(f"[router {time.strftime('%H:%M:%S')}]", *a, flush=True)


def sh(*cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except Exception as e:
        return 1, str(e)


def backend_containers():
    """Every running backend container (ninfer, llama.cpp or Bonsai), newest first."""
    rc, out = sh("docker", "ps", "--format", "{{.ID}} {{.Image}}")
    if rc != 0:
        return []
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in (IMAGE, LLAMA_IMAGE, BONSAI_IMAGE):
            found.append(parts[0])
    return found


def backend_container():
    """Newest backend container from any stack (ninfer, llama or bonsai)."""
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


def stop_backend():
    """Stop EVERY backend container: one model up means the other goes down.

    Stopping only the newest would leave a second resident backend holding
    VRAM after a manual launch, so the pre-flight gate must see a clean card.
    """
    ids = backend_containers()
    if ids:
        log("stopping backend", *ids)
        sh("docker", "stop", *ids, timeout=120)
    state["profile"] = None
    state["profile_mmproj"] = False


SERVE_LLAMA = os.environ.get(
    "LLAMA_SERVE_SH", os.path.join(stack_dir("gguf-stack"), "serve-llama.sh"))
SERVE_BONSAI = os.environ.get(
    "BONSAI_SERVE_SH",
    os.path.join(stack_dir("bonsai-stack"), "serve-bonsai.sh"))


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
    if profile in PROFILES:
        return [SERVE, PROFILE_MODELS.get(profile, MODEL_FILE),
                "--profile", profile, "--port", str(BACKEND_PORT)]
    return None


def start_backend(profile, want_mmproj=False):
    env = dict(os.environ)
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
                return True
        except Exception:
            pass
        time.sleep(3)
    log(f"profile {profile} failed to listen in {BOOT_TIMEOUT}s")
    return False


EFFORT_SUFFIXES = ("none", "low", "medium", "high", "xhigh", "minimal")

# Backend family -> {tier: (template reasoning_effort, thinking budget)}.
#
# The Qwen3.8 template only accepts xhigh|medium|low; "medium" adds no
# instruction at all, so low/medium differ only by budget. high maps to xhigh
# (the template rejects "high"); xhigh stays unbounded. minimal is an OpenAI
# alias for low. Budgets are per thinking block and only injected for
# llama.cpp backends (GGUF_FILES + BONSAI_FILES): ninfer-serve has no budget
# field, hence POLICY_UNCAPPED.
#
# The budgets are SAFETY NETS, not rate limiters: sized above the reasoning
# the model actually uses, so they only cut pathological turns. Measured with
# the cap removed (tokens, realistic prompts): Qwen 675-2424, Bonsai
# 730-4242, trivial prompts ~30-55. At 512 -- well below both -- Qwen answered
# 54% of a verifiable problem correctly vs 100% uncapped (15/28 vs 26/26),
# Bonsai 72% vs 100% (13/18 vs 16/16). Three candidate transition messages
# were indistinguishable from each other, so the cut itself, not its wording,
# is what costs accuracy. Bonsai sits ~1.7x above Qwen on the same prompt,
# hence the two tables.
EFFORT_POLICY = {
    "gguf": {
        "none": ("none", None),
        "low": ("medium", 2048),
        "medium": ("medium", 3072),
        "high": ("xhigh", 8192),
        "xhigh": ("xhigh", None),
    },
    "bonsai": {
        "none": ("none", None),
        "low": ("medium", 4096),
        "medium": ("medium", 6144),
        "high": ("xhigh", 8192),
        "xhigh": ("xhigh", None),
    },
}
# Backends with no budget field at all (ninfer): template hint only.
POLICY_UNCAPPED = {"none": ("none", None), "low": ("medium", None),
                   "medium": ("medium", None), "high": ("xhigh", None),
                   "xhigh": ("xhigh", None)}


def effort_policy(family):
    return EFFORT_POLICY.get(family, POLICY_UNCAPPED)
EFFORT_ALIASES = {"minimal": "low"}
BUDGET_KEYS = ("reasoning_budget_tokens", "thinking_budget_tokens")

# Injected right before the forced end-of-thinking tag when the budget runs
# out. Without it llama.cpp forces a bare </think> mid-sentence; Bonsai then
# frequently emits EOS and the turn ends with no answer at all.
BUDGET_MESSAGE = "\n\nOkay, I have thought enough. Time to write the final answer."


def parse_model(model):
    """Split '<base>[-effort]' into (profile, effort|None).

    Effort is the trailing '-<effort>' segment (xhigh matched before high).
    The '-max' tier marker is not an effort: 'qwen3.8-27b-max' has no suffix,
    'qwen3.8-27b-max-medium' is tier max + effort medium.
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
    if "vision" in base or "35b" in base:
        return "vision", effort
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
    if base.startswith("qwen3.8-27b-max") or "192k" in base:
        return "max", effort
    if "256k" in base:
        return "ultra", effort
    if "64k" in base:
        return "c1", effort
    return "balanced", effort


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

    Media on a GGUF tier wants mmproj on that SAME tier (lazy upgrade), not a
    tier switch. Media on ninfer text forces 35B vision.
    """
    profile, suffix_effort = parse_model(decoded.get("model"))
    notes = []
    want_mmproj = profile in ("gguf-vision", "bonsai-vision")
    if has_media(decoded):
        if profile in LLAMA_ALL:
            # Same tier, lazy encoder: reboot it with mmproj instead of
            # switching tiers (prefix cache is lost either way; tier stays).
            want_mmproj = True
            notes.append("media wants mmproj")
        elif profile not in VISION_CAPABLE:
            # Image/video input requires a vision-capable backend; ninfer
            # text tiers go 35B vision, otherwise the server 400s
            # vision_disabled.
            profile = "vision"
            notes.append("media forces vision")
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
    """Rewrite one request to the tier's template hint + hard budget.

    Explicit client thinking wins: a disabled thinking block is never
    re-enabled, and an explicit budget is never overwritten. `family` selects
    the budget table; backends without a budget field fall back to
    POLICY_UNCAPPED (template hint only).
    """
    template, budget = effort_policy(family)[effort]
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
    - Budgets only reach llama.cpp backends (final_profile in LLAMA_ALL);
      ninfer has no reasoning_budget_tokens equivalent.
    - Drops non-none effort on 35B vision: that template supports thinking
      on/off only; low/medium/xhigh 400 (reasoning_effort_not_supported).
    Explicit client values always win except on 35B vision, where only none
    survives.
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
    if profile == "vision":
        if path.startswith("/v1/chat/completions"):
            if (suffix_effort == "none"
                    and "reasoning_effort" not in decoded):
                decoded["reasoning_effort"] = "none"
                modified = True
                notes.append("injected none")
            if decoded.get("reasoning_effort") not in (None, "none"):
                notes.append(
                    f"vision dropped {decoded.pop('reasoning_effort')}")
                modified = True
            for key in BUDGET_KEYS:
                if key in decoded:
                    decoded.pop(key)
                    modified = True
                    notes.append(f"vision dropped {key}")
    else:
        effort, origin = resolve_effort(decoded, suffix_effort)
        meta["effort"] = effort
        meta["origin"] = origin
        if effort and effort not in POLICY_UNCAPPED:
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
            if (path.startswith("/v1/messages")
                    and profile not in LLAMA_ALL):
                notes.append(f"effort {effort} skipped for ninfer")
            else:
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
        {"id": "qwen3.8-27b-64k", "object": "model", "owned_by": "ninfer",
         "name": "qwen3.8-27b-64k", "context_window": 65536,
         "max_model_len": 65536, "modalities": {"vision": False}},
        {"id": "qwen3.8-27b", "object": "model", "owned_by": "ninfer",
         "name": "qwen3.8-27b", "context_window": 131072,
         "max_model_len": 131072, "modalities": {"vision": False}},
        {"id": "qwen3.8-27b-max", "object": "model", "owned_by": "ninfer",
         "name": "qwen3.8-27b-max", "context_window": 196608,
         "max_model_len": 196608, "modalities": {"vision": False}},
        {"id": "qwen3.6-35b-vision", "object": "model", "owned_by": "ninfer",
         "name": "qwen3.6-35b-vision", "context_window": 16384,
         "max_model_len": 16384, "modalities": {"vision": True}},
        {"id": "qwen3.8-27b-256k", "object": "model", "owned_by": "ninfer",
         "name": "qwen3.8-27b-256k", "context_window": 262144,
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
    # ninfer-serve tier: --max-context <n> (space or '=' form).
    win = None
    for i, t in enumerate(toks):
        if t.startswith("--max-context"):
            if "=" in t:
                win = t.split("=", 1)[1]
            elif i + 1 < len(toks):
                win = toks[i + 1]
            break
    if win is not None:
        return {"65536": "c1", "131072": "balanced", "196608": "max",
                "262144": "ultra", "32768": "vision"}.get(win)
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
    if not os.path.isfile(MODEL_FILE):
        raise SystemExit(f"model not found: {MODEL_FILE}")
    # Adopt an already-running backend instead of rebooting it: infer the
    # profile from the container's server args (llama --alias / ninfer
    # --max-context).
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
