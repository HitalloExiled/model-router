#!/usr/bin/env python3
"""Cross-process slot persistence probe (llama-server `--slot-save-path`).

The question this answers is not "does POST /slots/.../save return 200" but
"does a fresh process actually reuse the restored KV instead of re-prefilling
the prompt". Restore has been reported to answer OK and still leave an empty
cache, so the only trustworthy signal is `timings.cache_n` on the next request.

The prompt is deterministic, so `prime` in one process and `measure` in the
next are the same tokens.

Typical run (backend restarted in between):

  scripts/slot-persist-probe.py prime   --file probe.bin   # saves, prints slot
  # ... docker stop / start ...
  scripts/slot-persist-probe.py restore --slot 2 --file probe.bin
  scripts/slot-persist-probe.py measure                    # warm: cache_n high
  scripts/slot-persist-probe.py erase --slot 2
  scripts/slot-persist-probe.py measure                    # cold: cache_n 0

Exit codes: 0 answered, 2 transport/protocol failure.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

WORDS = ("amber basil cedar delta ember fable garnet harbor indigo jasper "
         "kestrel lumen meadow nectar onyx pepper quartz raven saffron "
         "thistle umber violet willow xenon yarrow zephyr").split()


def prompt(n_words):
    # Fixed salt: identical bytes on every invocation, on purpose.
    return "slot persistence probe\n" + " ".join(
        f"{WORDS[i % len(WORDS)]}sp{i}" for i in range(n_words))


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else {})


def complete(base, n_words):
    t0 = time.monotonic()
    _, res = post(base + "/completion", {
        "prompt": prompt(n_words),
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": True,
    })
    wall = (time.monotonic() - t0) * 1000
    timings = res.get("timings") or {}
    total = res.get("tokens_evaluated")
    cached = timings.get("cache_n")
    return {
        "slot": res.get("id_slot"),
        "total": total,
        "cached": cached,
        "evaluated": (total - cached) if total is not None and cached is not None else None,
        "prefill_ms": timings.get("prompt_ms"),
        "wall_ms": round(wall),
    }


def slot_action(base, action, slot, filename=None):
    body = {"filename": filename} if filename else {}
    t0 = time.monotonic()
    status, res = post(base + f"/slots/{slot}?action={action}", body)
    return status, res, (time.monotonic() - t0) * 1000


def show(label, m):
    print(f"{label:<12} slot={m['slot']} total={m['total']} "
          f"cache_n={m['cached']} evaluated={m['evaluated']} "
          f"prefill={m['prefill_ms']:.0f} ms wall={m['wall_ms']} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["prime", "measure", "restore", "erase", "cycle"])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--file", default="probe-slot.bin")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--words", type=int, default=4000)
    args = ap.parse_args()
    base = args.url

    if args.command == "measure":
        show("measure", complete(base, args.words))
        return 0

    if args.command == "prime":
        m = complete(base, args.words)
        show("prime", m)
        if m["slot"] is None:
            print("no slot reported; cannot save", file=sys.stderr)
            return 2
        status, res, ms = slot_action(base, "save", m["slot"], args.file)
        print(f"save        slot={m['slot']} file={args.file} status={status} "
              f"{ms:.0f} ms {json.dumps(res)[:200]}")
        return 0 if status == 200 else 2

    if args.command == "restore":
        status, res, ms = slot_action(base, "restore", args.slot, args.file)
        print(f"restore     slot={args.slot} file={args.file} status={status} "
              f"{ms:.0f} ms {json.dumps(res)[:200]}")
        return 0 if status == 200 else 2

    if args.command == "erase":
        status, res, ms = slot_action(base, "erase", args.slot)
        print(f"erase       slot={args.slot} status={status} "
              f"{ms:.0f} ms {json.dumps(res)[:120]}")
        return 0 if status == 200 else 2

    # cycle: everything inside one process, as a sanity check before a restart
    m = complete(base, args.words)
    show("prime", m)
    slot = m["slot"]
    status, res, ms = slot_action(base, "save", slot, args.file)
    print(f"save        slot={slot} status={status} {ms:.0f} ms")
    status, res, ms = slot_action(base, "erase", slot)
    print(f"erase       slot={slot} status={status} {ms:.0f} ms")
    show("after-erase", complete(base, args.words))
    status, res, ms = slot_action(base, "restore", slot, args.file)
    print(f"restore     slot={slot} status={status} {ms:.0f} ms")
    show("after-restore", complete(base, args.words))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"probe failed: {e!r}", file=sys.stderr)
        sys.exit(2)
