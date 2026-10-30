#!/usr/bin/env python3
"""Measure prompt-processing (prefill) throughput of the resident backend.

Talking to /completion with n_predict=1 and cache_prompt=false gives the
server's own timings.prompt_ms/prompt_per_second, so a "prefill feels slow"
report can be split into the two things that actually happen:

  * fixed first-batch cost -- ~100 ms on both stacks, which at 90 tokens
    reads as ~850 tok/s no matter how fast the GPU is;
  * real prefill rate -- ~3.2k tok/s for a 27B on the 4090 once the prompt
    is long enough to amortize that.

Usage:
  scripts/prefill-probe.py                        # 128,1024,4096,16384
  scripts/prefill-probe.py --sizes 8192,32768
  scripts/prefill-probe.py --backend http://127.0.0.1:8080

Reads /props first so the model and slot count are part of the output.
Exit codes: 0 measured, 2 backend unreachable, 3 /completion failed.
"""
import argparse
import json
import sys
import time
import urllib.request

FILLER = ("The quick brown fox jumps over the lazy dog while the router serves "
          "several concurrent slots and measures prefill throughput. ")


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post_json(url, body, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8080")
    ap.add_argument("--sizes", default="128,1024,4096,16384",
                    help="approximate prompt sizes in tokens (comma separated)")
    args = ap.parse_args()
    base = args.backend.rstrip("/")

    try:
        props = get_json(base + "/props")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot read {base}/props: {e}")
        return 2

    model = str(props.get("model_path", "?")).split("/")[-1]
    slots = props.get("total_slots") or props.get("n_parallel") or 1
    print(f"model: {model}  slots: {slots}")

    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        # ~4 characters per token keeps prompt_n close to the target.
        prompt = (FILLER * (size * 4 // len(FILLER) + 1))[: size * 4]
        t0 = time.time()
        try:
            out = post_json(base + "/completion",
                            {"prompt": prompt, "n_predict": 1, "temperature": 0,
                             "top_k": 1, "cache_prompt": False})
        except Exception as e:  # noqa: BLE001
            print(f"target {size:>6} tok | FAIL /completion: {e}")
            return 3
        wall = time.time() - t0
        t = out.get("timings") or {}
        print(f"target {size:>6} tok | prompt_n={t.get('prompt_n', '?'):>6} "
              f"prompt_ms={t.get('prompt_ms', 0):>9.1f} "
              f"pp={t.get('prompt_per_second', 0):>8.1f} tok/s "
              f"| wall={wall:>6.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
