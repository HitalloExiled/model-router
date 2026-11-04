#!/usr/bin/env python3
"""A/B the server-side `n_cache_reuse` (KV shifting) against a diverged prefix.

The interesting case is not "append a message" (llama-server's common-prefix
trim already handles that for free). It is a prompt whose head changed while
the tail stayed put: a tool list that gained an entry, a system prompt edited
above the conversation, a message inserted mid-history. Without KV shifting
the server trims to the common prefix and re-prefills everything after it.

This probe primes a slot with P0, then asks for P1 (P0 with a short block
inserted near the top) and reports how many tokens the server actually had to
evaluate. It runs against the BACKEND (default :8080), never the router, and
does not need a restart: `n_cache_reuse` is a per-request field.

Usage:
  scripts/cache-reuse-probe.py                 # A/B 0 vs 256
  scripts/cache-reuse-probe.py --reuse 0 128 512
  scripts/cache-reuse-probe.py --filler 6000 --insert 40

Exit codes: 0 server answered, 2 transport/protocol failure.
No third-party deps.
"""
import argparse
import json
import sys
import time
import urllib.request

# Deterministic filler. Words are distinct so the tokenizer cannot merge the
# seam between the inserted block and the tail in a way that depends on the
# insertion, which would hide a shift behind a token mismatch.
WORDS = ("amber basil cedar delta ember fable garnet harbor indigo jasper "
         "kestrel lumen meadow nectar onyx pepper quartz raven saffron "
         "thistle umber violet willow xenon yarrow zephyr").split()


def filler(n_words):
    return " ".join(f"{WORDS[i % len(WORDS)]}{i}" for i in range(n_words))


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run(base_url, prompt, reuse, n_predict=1, slot=None):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
        "n_cache_reuse": reuse,
        "timings_per_token": False,
    }
    if slot is not None:
        payload["id_slot"] = slot
    t0 = time.monotonic()
    res = post(base_url + "/completion", payload)
    wall = time.monotonic() - t0
    timings = res.get("timings") or {}
    total = res.get("tokens_evaluated")
    cached = timings.get("cache_n")
    return {
        "reuse": reuse,
        "slot": res.get("id_slot"),
        "total": total,
        "cached": cached,
        "evaluated": (total - cached) if total is not None and cached is not None else None,
        "wall_ms": round(wall * 1000),
        "prefill_ms": timings.get("prompt_ms"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--filler", type=int, default=3000,
                    help="words in the shared tail (default 3000)")
    ap.add_argument("--insert", type=int, default=40,
                    help="words inserted near the top (default 40)")
    ap.add_argument("--reuse", type=int, nargs="+", default=[0, 256])
    ap.add_argument("--slot", type=int, default=None,
                    help="pin every request to this slot (needed to test KV "
                         "shifting: the server's selector is prefix-only and "
                         "would send a head-diverged prompt to a cold slot)")
    args = ap.parse_args()

    head = "system: you are a careful assistant.\n"
    tail = filler(args.filler)
    p0 = head + tail
    p1 = head + "note: an extra tool was attached.\n" + tail

    print(f"backend {args.url} · tail={args.filler} words · "
          f"insert={args.insert} words\n")
    results = []

    if args.slot is None:
        prime = run(args.url, p0, 0)
        slot = prime["slot"]
        print(f"prime     P0 -> slot {slot}")
    else:
        slot = args.slot

    # Baseline: identical prompt twice, pinned, so the probe measures the cache
    # and not the selector.
    run(args.url, p0, 0, slot=slot)
    warm = run(args.url, p0, 0, slot=slot)
    print(f"baseline  same prompt 2x      total={warm['total']:>5} "
          f"cache_n={warm['cached']:>5} evaluated={warm['evaluated']:>5} "
          f"slot={warm['slot']}")
    results.append(("baseline", warm))

    for reuse in args.reuse:
        # Re-prime so every variant starts from the same resident prompt.
        run(args.url, p0, 0, slot=slot)
        hit = run(args.url, p1, reuse, slot=slot)
        print(f"insert    n_cache_reuse={reuse:<4}   total={hit['total']:>5} "
              f"cache_n={hit['cached']:>5} evaluated={hit['evaluated']:>5} "
              f"slot={hit['slot']} prefill={hit['prefill_ms']:.0f} ms")
        results.append((f"reuse={reuse}", hit))

    base = results[1][1]
    best = min((r for k, r in results if k.startswith("reuse=")),
               key=lambda r: r["evaluated"] or 0)
    saved = (base["evaluated"] or 0) - (best["evaluated"] or 0)
    if base is not best and saved > 0:
        print(f"\nbest: n_cache_reuse={best['reuse']} evaluates {saved} fewer "
              f"tokens ({(best['evaluated'] or 0) / max(base['evaluated'] or 1, 1):.0%} "
              f"of the no-reuse prefill)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"probe failed: {e!r}", file=sys.stderr)
        sys.exit(2)
