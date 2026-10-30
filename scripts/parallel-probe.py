#!/usr/bin/env python3
"""Measure whether a router tier really runs requests concurrently.

llama-server queues a request when every slot is busy, so "it feels slow"
can mean "the resident backend has one slot". This probe distinguishes the
two failure modes empirically:

  * peak slots busy (from the backend /slots endpoint)
  * wall-clock overlap of the streaming intervals
  * per-request TTFT

It talks to the ROUTER (default :18089) so the model name selects the tier,
and to the BACKEND directly (default :8080) for introspection.

Usage:
  scripts/parallel-probe.py --model bonsai-2-27b-quad128 --requests 4
  scripts/parallel-probe.py --model qwen3.8-gguf-duo128 --requests 3 --tokens 64

Exit codes: 0 all requests ran concurrently up to the slot count,
            1 queueing detected (peak < expected),
            2 backend unreachable / slot introspection unavailable.
No third-party deps.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

BACKEND_IMAGES = (
    "ghcr.io/ggml-org/llama.cpp:server-cuda",
    "bonsai-llama-4090",
    "ninfer-4090",
)


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def stream_chat(base, model, prompt, max_tokens, out):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        # Thinking off: we measure scheduling, not the reasoning budget.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    out.update(ttft=None, end=None, chars=0, ok=False, err=None)
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                choices = j.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if out["ttft"] is None:
                        out["ttft"] = time.time() - t0
                    out["chars"] += len(piece)
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 - probe reports, never raises
        out["err"] = str(e)
    out["end"] = time.time() - t0


def poll_slots(backend, stop, samples):
    url = backend.rstrip("/") + "/slots"
    while not stop.is_set():
        try:
            data = get_json(url, timeout=5)
            if isinstance(data, dict):
                data = data.get("slots", [])
            samples.append(sum(1 for s in data if s.get("is_processing")))
        except Exception:  # noqa: BLE001
            samples.append(-1)
        time.sleep(0.2)


def max_overlap(intervals):
    """Largest number of intervals sharing an instant."""
    events = []
    for a, b in intervals:
        events.append((a, 1))
        events.append((b, -1))
    events.sort()
    cur = best = 0
    for _t, d in events:
        cur += d
        best = max(best, cur)
    return best


def backend_containers():
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}} {{.Image}}"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return []
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and any(i in parts[1] for i in BACKEND_IMAGES):
            found.append(parts[0])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="http://127.0.0.1:18089/v1")
    ap.add_argument("--backend", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="bonsai-2-27b-quad128")
    ap.add_argument("--requests", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=128)
    # Assert a specific concurrency floor instead of "as many as the backend
    # happens to have". A 1-slot backend would otherwise PASS trivially.
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    if not args.skip_warmup:
        warm = {}
        stream_chat(args.router, args.model,
                    "warmup: reply with the single word ok", 8, warm)
        if not warm["ok"]:
            print(f"FAIL warmup: {warm['err']}")
            return 2

    props = get_json(args.backend.rstrip("/") + "/props")
    slots = props.get("total_slots") or props.get("n_parallel") or 1
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx") \
        or props.get("n_ctx")
    model_path = str(props.get("model_path", "?"))
    print(f"backend: {model_path.split('/')[-1]}")
    print(f"slots={slots}  n_ctx_per_slot={n_ctx}  "
          f"containers={len(backend_containers())}")

    results = [{} for _ in range(args.requests)]
    stop = threading.Event()
    samples = []
    poller = threading.Thread(target=poll_slots,
                              args=(args.backend, stop, samples), daemon=True)
    poller.start()

    threads = []
    nonce = uuid.uuid4().hex[:8]
    t_start = time.time()
    for i in range(args.requests):
        # Distinct leading token per request: a shared prefix would let the
        # server reuse cached KV and hide the scheduling behaviour.
        prompt = (f"Request {nonce}-{i}: list the numbers from 1 to 40, "
                  f"one per line, then say done.")
        th = threading.Thread(target=stream_chat,
                              args=(args.router, args.model, prompt,
                                    args.tokens, results[i]))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.time() - t_start
    stop.set()
    poller.join(timeout=5)

    # Re-read: the router may have switched tiers for this very request
    # (slot-aware reuse), so the pre-run snapshot can describe the old backend.
    try:
        props_after = get_json(args.backend.rstrip("/") + "/props")
        slots_after = props_after.get("total_slots") \
            or props_after.get("n_parallel") or slots
        model_after = str(props_after.get("model_path", "?")).split("/")[-1]
    except Exception:  # noqa: BLE001
        slots_after, model_after = slots, model_path.split("/")[-1]
    if int(slots_after) != int(slots) or model_after != model_path.split("/")[-1]:
        print(f"switched during run: slots {slots}->{slots_after}, "
              f"model {model_path.split('/')[-1]}->{model_after}")
    slots = slots_after

    good = [r for r in results if r.get("ok")]
    intervals = [(r["ttft"], r["end"]) for r in good
                 if r.get("ttft") is not None]
    overlap = max_overlap(intervals) if intervals else 0
    peak = max([s for s in samples if s >= 0], default=-1)

    print(f"\n{'#':>2} {'ttft_s':>8} {'end_s':>8} {'chars':>6}  status")
    for i, r in enumerate(results):
        if r.get("ok"):
            print(f"{i:>2} {r['ttft']:>8.2f} {r['end']:>8.2f} "
                  f"{r['chars']:>6}  ok")
        else:
            print(f"{i:>2} {'-':>8} {'-':>8} {'-':>6}  {r.get('err')}")
    print(f"\nwall={wall:.2f}s  overlap={overlap}  peak_slots_busy={peak}")
    if not good:
        print("FAIL: no request completed")
        return 2

    expected = args.expect if args.expect is not None \
        else min(args.requests, int(slots))
    observed = max(overlap, peak, 0)
    if observed >= expected:
        print(f"PASS: {observed} concurrent stream(s), expected {expected} "
              f"({args.requests} requests / {slots} slots)")
        return 0
    print(f"FAIL: only {observed} concurrent stream(s), expected {expected} "
          f"-- the backend is QUEUEING")
    if expected > 1 and slots == 1:
        print("hint: the resident tier has a single slot; request a "
              "duo/quint/octo tier or raise --parallel.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
