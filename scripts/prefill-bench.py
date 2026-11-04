#!/usr/bin/env python3
"""Measure raw prefill throughput on a llama-server backend.

`cache_prompt: false` makes every run a real prefill, so the number is not
distorted by a warm slot. Each run gets a distinct salt, and repeats are
reported with their median: prefill is short enough that a single noisy run
(GPU clock ramp, another process) would otherwise decide an A/B.

Usage:
  scripts/prefill-bench.py --words 4000 --runs 3 --label ub512
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

WORDS = ("amber basil cedar delta ember fable garnet harbor indigo jasper "
         "kestrel lumen meadow nectar onyx pepper quartz raven saffron "
         "thistle umber violet willow xenon yarrow zephyr").split()


def filler(n_words, salt):
    return " ".join(f"{WORDS[i % len(WORDS)]}{salt}x{i}" for i in range(n_words))


def prefill(url, words, salt):
    prompt = (f"run {salt}\n" + filler(words, salt))
    t0 = time.monotonic()
    req = urllib.request.Request(
        url + "/completion",
        data=json.dumps({
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "cache_prompt": False,
            "n_cache_reuse": 0,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.load(r)
    wall_ms = (time.monotonic() - t0) * 1000
    t = res.get("timings") or {}
    return {
        "tokens": res.get("tokens_evaluated"),
        "prompt_ms": t.get("prompt_ms"),
        "tok_s": t.get("prompt_per_second"),
        "wall_ms": round(wall_ms),
        "slot": res.get("id_slot"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--words", type=int, default=4000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = []
    for i in range(args.runs):
        rows.append(prefill(args.url, args.words, f"{args.label}{i}"))
        r = rows[-1]
        print(f"  {args.label} run{i}: {r['tokens']} tokens "
              f"{r['prompt_ms']:.0f} ms -> {r['tok_s']:.0f} tok/s "
              f"(wall {r['wall_ms']} ms, slot {r['slot']})")
    rates = [r["tok_s"] for r in rows]
    med = statistics.median(rates)
    print(f"  {args.label} median: {med:.0f} tok/s "
          f"(min {min(rates):.0f}, max {max(rates):.0f})")
    return med


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"bench failed: {e!r}", file=sys.stderr)
        sys.exit(2)
