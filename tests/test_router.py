"""Offline unit tests for the model-router request handling.

Covers parse_model (model name -> profile + effort), normalize_chat_body
(effort injection, vision normalization, media forcing) and
effective_profile (backend reuse). No GPU, docker, or network needed.
"""
import importlib.util
import json
import os
import sys

ROUTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "model-router.py")


def load():
    spec = importlib.util.spec_from_file_location("nr", ROUTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(label)
        print(f"FAIL {label}: got={got!r} want={want!r}")
    else:
        print(f"PASS {label}")


def main():
    nr = load()

    for name, want in {
        "qwen3.8-27b": ("balanced", None),
        "qwen3.8-27b-low": ("balanced", "low"),
        "qwen3.8-27b-64k-xhigh": ("c1", "xhigh"),
        "qwen3.8-27b-max": ("max", None),
        "qwen3.8-27b-max-medium": ("max", "medium"),
        "qwen3.8-27b-192k": ("max", None),
        "qwen3.8-27b-192k-low": ("max", "low"),
        "qwen3.8-27b-256k-xhigh": ("ultra", "xhigh"),
        "qwen3.8-27b-256k": ("ultra", None),
        "qwen3.8-27b-128k-medium": ("balanced", "medium"),
        "qwen3.6-35b-vision": ("vision", None),
        "qwen3.6-35b-vision-low": ("vision", "low"),
        "qwen3.8-gguf-speed": ("gguf-speed", None),
        "qwen3.8-gguf-range-xhigh": ("gguf-range", "xhigh"),
        "qwen3.8-gguf-max": ("gguf-max", None),
        "qwen3.8-gguf-duo": ("gguf-duo", None),
        "qwen3.8-gguf-duo128-low": ("gguf-duo128", "low"),
        "qwen3.8-gguf-quad": ("gguf-quad", None),
    }.items():
        check(f"parse:{name}", nr.parse_model(name), want)

    P = "/v1/chat/completions"
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}

    def norm(model, content="hi", effort=None):
        d = {"model": model, "messages": [{"role": "user", "content": content}]}
        if effort:
            d["reasoning_effort"] = effort
        prof, new_body, _note, _mm = nr.normalize_chat_body(P, json.dumps(d).encode())
        return prof, json.loads(new_body).get("reasoning_effort", "<ABSENT>")

    for label, model, content, sent, want in [
        ("vision-suffix-dropped", "qwen3.6-35b-vision-low", "hi", None,
         ("vision", "<ABSENT>")),
        ("vision-explicit-low-dropped", "qwen3.6-35b-vision", "hi", "low",
         ("vision", "<ABSENT>")),
        ("vision-none-kept", "qwen3.6-35b-vision", "hi", "none",
         ("vision", "none")),
        ("balanced-inject-low", "qwen3.8-27b-128k-low", "hi", None,
         ("balanced", "low")),
        ("explicit-wins", "qwen3.8-27b-128k-low", "hi", "medium",
         ("balanced", "medium")),
        ("media-forces-vision", "qwen3.8-27b-128k", [img], None,
         ("vision", "<ABSENT>")),
        ("plain-untouched", "qwen3.8-27b-256k", "hi", None,
         ("ultra", "<ABSENT>")),
    ]:
        check(f"norm:{label}", norm(model, content, sent), want)

    E = nr.effective_profile
    for label, w, c, e, m, want in [
        ("same", "balanced", "balanced", 1000, False, "balanced"),
        ("reuse-smaller", "ultra", "balanced", 1000, False, "balanced"),
        ("overflow-switches", "ultra", "balanced", 120000, False, "ultra"),
        ("reuse-bigger", "balanced", "ultra", 1000, False, "ultra"),
        ("nothing-running", "balanced", None, 1000, False, "balanced"),
        ("media-to-vision", "vision", "balanced", 1000, True, "vision"),
        ("vision-fits-stays", "balanced", "vision", 1000, False, "vision"),
        ("vision-overflow-leaves", "balanced", "vision", 16000, False, "balanced"),
    ]:
        check(f"reuse:{label}", E(w, c, e, m), want)

    # Encoder (mmproj) state must be tracked separately from the profile name,
    # or a media request reboots the same GGUF backend on every call.
    S = nr.same_backend
    for label, cn, cm, p, wm, want in [
        ("exact-text", "balanced", False, "balanced", False, True),
        ("encoder-added-reboots", "gguf-max", False, "gguf-max", True, False),
        ("encoder-kept-reuses", "gguf-max", True, "gguf-max", True, True),
        ("encoder-removed-reboots", "gguf-max", True, "gguf-max", False, False),
        ("tier-differs", "balanced", False, "ultra", False, False),
        ("nothing-running", None, False, "balanced", False, False),
    ]:
        check(f"same:{label}", S(cn, cm, p, wm), want)

    # Adoption must recognize a live container from its server args, else the
    # router reboots a healthy backend on startup.
    I = nr.infer_profile
    for label, args, want in [
        ("llama-alias-single192",
         "-m /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --alias "
         "qwen3.8-gguf-single192 -c 196608 --parallel 1 --mmproj /m/mmproj.gguf",
         "gguf-single192"),
        ("llama-alias-max", "--alias qwen3.8-gguf-max -c 262144", "gguf-max"),
        ("llama-no-alias-parallel4", "-m x.gguf --parallel 4", "gguf-quad"),
        ("llama-no-alias-parallel2-128",
         "-m x.gguf --parallel 2 -c 131072", "gguf-duo"),
        ("llama-no-alias-parallel2-256",
         "-m x.gguf --parallel 2 -c 262144", "gguf-duo128"),
        ("ninfer-131072", "--max-context 131072 --kv-dtype rk8v4", "balanced"),
        ("ninfer-eq-262144", "--max-context=262144", "ultra"),
        ("unknown", "--foo bar", None),
        ("empty", "", None),
    ]:
        check(f"adopt:{label}", I(args), want)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES")
        sys.exit(1)
    print("\nrouter unit tests passed.")


if __name__ == "__main__":
    main()
