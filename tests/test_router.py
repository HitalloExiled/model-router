"""Offline unit tests for the model-router request handling.

Covers parse_model (model name -> profile + effort), normalize_chat_body
(effort policy, budgets, vision normalization, media forcing),
/v1/messages effort injection and effective_profile (backend reuse).
No GPU, docker, or network needed.
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
        "qwen3.8-27b-128k-minimal": ("balanced", "minimal"),
        "qwen3.6-35b-vision": ("vision", None),
        "qwen3.6-35b-vision-low": ("vision", "low"),
        "qwen3.8-gguf-speed": ("gguf-speed", None),
        "qwen3.8-gguf-range-xhigh": ("gguf-range", "xhigh"),
        "qwen3.8-gguf-max": ("gguf-max", None),
        "qwen3.8-gguf-duo": ("gguf-duo", None),
        "qwen3.8-gguf-duo128-low": ("gguf-duo128", "low"),
        "qwen3.8-gguf-quad": ("gguf-quad", None),
        "bonsai-2-27b-max": ("bonsai-max", None),
        "bonsai-2-27b-quad128-low": ("bonsai-quad128", "low"),
        "bonsai-2-27b-octo64-xhigh": ("bonsai-octo64", "xhigh"),
        "bonsai-2-27b-duo128": ("bonsai-duo128", None),
        "bonsai-2-27b-vision": ("bonsai-vision", None),
        "bonsai-quad128": ("bonsai-quad128", None),
    }.items():
        check(f"parse:{name}", nr.parse_model(name), want)

    P = "/v1/chat/completions"
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}

    def norm(model, content="hi", effort=None, extra=None,
             final_profile=None, path=P):
        d = {"model": model, "messages": [{"role": "user", "content": content}]}
        if effort:
            d["reasoning_effort"] = effort
        if extra:
            d.update(extra)
        prof, new_body, _note, _mm, meta = nr.normalize_chat_body(
            path, json.dumps(d).encode(), final_profile=final_profile)
        return prof, json.loads(new_body), meta

    def chat(model, content="hi", effort=None, extra=None,
             final_profile=None):
        prof, out, _meta = norm(model, content, effort, extra, final_profile)
        return (prof,
                out.get("reasoning_effort", "<ABSENT>"),
                out.get("reasoning_budget_tokens", "<ABSENT>"))

    for label, args, want in [
        ("vision-suffix-dropped", ("qwen3.6-35b-vision-low",),
         ("vision", "<ABSENT>", "<ABSENT>")),
        ("vision-explicit-low-dropped", ("qwen3.6-35b-vision", "hi", "low"),
         ("vision", "<ABSENT>", "<ABSENT>")),
        ("vision-none-kept", ("qwen3.6-35b-vision", "hi", "none"),
         ("vision", "none", "<ABSENT>")),
        ("vision-budget-dropped",
         ("qwen3.6-35b-vision", "hi", "low",
          {"reasoning_budget_tokens": 4096}),
         ("vision", "<ABSENT>", "<ABSENT>")),
        ("low-is-budgeted", ("qwen3.8-gguf-max-low",),
         ("gguf-max", "medium", 512)),
        ("medium-budget", ("qwen3.8-gguf-max-medium",),
         ("gguf-max", "medium", 2048)),
        ("high-maps-xhigh", ("qwen3.8-gguf-max-high",),
         ("gguf-max", "xhigh", 8192)),
        ("minimal-alias-low", ("qwen3.8-gguf-max-minimal",),
         ("gguf-max", "medium", 512)),
        ("xhigh-unbounded", ("qwen3.8-gguf-max-xhigh",),
         ("gguf-max", "xhigh", "<ABSENT>")),
        ("explicit-wins", ("qwen3.8-gguf-max-low", "hi", "medium"),
         ("gguf-max", "medium", 2048)),
        ("explicit-budget-wins",
         ("qwen3.8-gguf-max-low", "hi", None,
          {"reasoning_budget_tokens": 4096}),
         ("gguf-max", "medium", 4096)),
        ("client-thinking-off",
         ("qwen3.8-gguf-max-low", "hi", None,
          {"chat_template_kwargs": {"enable_thinking": False}}),
         ("gguf-max", "<ABSENT>", "<ABSENT>")),
        ("unsupported-dropped", ("qwen3.8-gguf-max", "hi", "max"),
         ("gguf-max", "<ABSENT>", "<ABSENT>")),
        ("ninfer-no-budget", ("qwen3.8-27b-128k-low",),
         ("balanced", "medium", "<ABSENT>")),
        ("reuse-ninfer-no-budget",
         ("qwen3.8-gguf-max-low", "hi", None, None, "balanced"),
         ("balanced", "medium", "<ABSENT>")),
        ("media-forces-vision", ("qwen3.8-27b-128k", [img]),
         ("vision", "<ABSENT>", "<ABSENT>")),
        ("plain-untouched", ("qwen3.8-27b-256k",),
         ("ultra", "<ABSENT>", "<ABSENT>")),
        # Ternary backend must get the same hard budgets as the GGUF one.
        ("bonsai-low-is-budgeted", ("bonsai-2-27b-quad128-low",),
         ("bonsai-quad128", "medium", 512)),
        ("bonsai-high-maps-xhigh", ("bonsai-2-27b-octo64-high",),
         ("bonsai-octo64", "xhigh", 8192)),
    ]:
        check(f"norm:{label}", chat(*args), want)

    M = "/v1/messages"

    def messages(model, effort=None, extra=None, final_profile=None):
        prof, out, meta = norm(model, effort=effort, extra=extra,
                               final_profile=final_profile, path=M)
        ctk = out.get("chat_template_kwargs") or {}
        thinking = out.get("thinking") or {}
        return (prof,
                ctk.get("reasoning_effort", "<ABSENT>"),
                ctk.get("enable_thinking", "<ABSENT>"),
                thinking.get("budget_tokens", "<ABSENT>"),
                meta.get("injected_budget", False))

    for label, args, want in [
        ("gguf-low", ("qwen3.8-gguf-max-low",),
         ("gguf-max", "medium", True, 512, True)),
        ("gguf-high", ("qwen3.8-gguf-max-high",),
         ("gguf-max", "xhigh", True, 8192, True)),
        ("gguf-none", ("qwen3.8-gguf-max-none",),
         ("gguf-max", "<ABSENT>", False, "<ABSENT>", False)),
        ("explicit-budget",
         ("qwen3.8-gguf-max-low", None,
          {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
         ("gguf-max", "medium", True, 4096, False)),
        ("thinking-disabled",
         ("qwen3.8-gguf-max-low", None,
          {"thinking": {"type": "disabled"}}),
         ("gguf-max", "<ABSENT>", "<ABSENT>", "<ABSENT>", False)),
        ("ninfer-skipped", ("qwen3.8-27b-128k-low",),
         ("balanced", "<ABSENT>", "<ABSENT>", "<ABSENT>", False)),
        ("reuse-ninfer-skipped",
         ("qwen3.8-gguf-max-low", None, None, "balanced"),
         ("balanced", "<ABSENT>", "<ABSENT>", "<ABSENT>", False)),
    ]:
        check(f"messages:{label}", messages(*args), want)

    # Bonsai media stays on its own tier with the lazy encoder, like GGUF.
    check("route:bonsai-media-lazy-mmproj",
          nr.route_request({"model": "bonsai-2-27b-octo64",
                            "messages": [{"role": "user", "content": [img]}]}),
          ("bonsai-octo64", None, True, ["media wants mmproj"]))

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
        # Slot-aware reuse: a resident 1-slot backend must not swallow a
        # multi-slot tier, and a wider tier still serves the narrower one.
        ("slots-block-reuse", "gguf-duo128", "gguf-range", 1000, False,
         "gguf-duo128"),
        ("slots-allow-reuse", "gguf-range", "gguf-duo128", 1000, False,
         "gguf-duo128"),
        ("quad-not-swallowed-by-duo", "gguf-quad", "gguf-duo128", 1000, False,
         "gguf-quad"),
        ("bonsai-doubles-gguf", "bonsai-quad128", "gguf-duo128", 1000, False,
         "bonsai-quad128"),
        ("bonsai-octo-not-swallowed", "bonsai-octo64", "bonsai-quad128", 1000,
         False, "bonsai-octo64"),
        ("bonsai-octo-serves-quad", "bonsai-quad128", "bonsai-octo64", 1000,
         False, "bonsai-octo64"),
        ("bonsai-max-reused-for-gguf", "gguf-range", "bonsai-max", 1000, False,
         "bonsai-max"),
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
        ("bonsai-alias", "--alias bonsai-quad128 -c 524288 --parallel 4",
         "bonsai-quad128"),
        ("bonsai-no-alias-octo",
         "-m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --parallel 8",
         "bonsai-octo64"),
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
