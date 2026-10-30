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
        "bonsai-2-27b-duo256": ("bonsai-duo256", None),
        "bonsai-2-27b-trio256-medium": ("bonsai-trio256", "medium"),
        "bonsai-2-27b-trio200": ("bonsai-trio200", None),
        "bonsai-2-27b-trio200-low": ("bonsai-trio200", "low"),
        "bonsai-2-27b-quad128-low": ("bonsai-quad128", "low"),
        "bonsai-2-27b-penta128": ("bonsai-penta128", None),
        "bonsai-2-27b-hexa128-low": ("bonsai-hexa128", "low"),
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
        # Bonsai opts out of the hard budget: the cap truncates its thinking.
        ("bonsai-low-no-cap", ("bonsai-2-27b-quad128-low",),
         ("bonsai-quad128", "medium", "<ABSENT>")),
        ("bonsai-high-maps-xhigh", ("bonsai-2-27b-octo64-high",),
         ("bonsai-octo64", "xhigh", "<ABSENT>")),
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
        ("bonsai-octo-not-swallowed", "bonsai-octo64", "bonsai-quad128", 1000,
         False, "bonsai-octo64"),
        ("bonsai-octo-serves-quad", "bonsai-quad128", "bonsai-octo64", 1000,
         False, "bonsai-octo64"),
        # Full-window tiers: 3 slots absorb 2, 2 must not swallow 3, and the
        # single-slot max tier yields to any multi-slot request.
        ("trio-not-swallowed-by-duo256", "bonsai-trio256", "bonsai-duo256",
         1000, False, "bonsai-trio256"),
        ("duo256-reused-by-trio", "bonsai-duo256", "bonsai-trio256", 1000,
         False, "bonsai-trio256"),
        ("max-not-swallowed-by-duo256", "bonsai-duo256", "bonsai-max", 1000,
         False, "bonsai-duo256"),
        ("duo256-not-swallowed-by-max", "bonsai-max", "bonsai-duo256", 1000,
         False, "bonsai-duo256"),
        # 128K ladder: 6 slots absorb 4/5, and a 200K prompt must leave a
        # 128K backend even though it has more slots than trio256.
        ("hexa-not-swallowed-by-quad", "bonsai-hexa128", "bonsai-quad128",
         1000, False, "bonsai-hexa128"),
        ("hexa-not-swallowed-by-penta", "bonsai-hexa128", "bonsai-penta128",
         1000, False, "bonsai-hexa128"),
        ("hexa-serves-quad", "bonsai-quad128", "bonsai-hexa128", 1000,
         False, "bonsai-hexa128"),
        ("hexa-serves-small-trio", "bonsai-trio256", "bonsai-hexa128", 1000,
         False, "bonsai-hexa128"),
        ("hexa-yields-200k-trio", "bonsai-trio256", "bonsai-hexa128", 200000,
         False, "bonsai-trio256"),
        # trio200 trades 56K of window per slot for ~4 GiB of VRAM headroom.
        # It still absorbs small trio256 requests, but a ~200K prompt needs
        # the full-window tier.
        ("trio200-reused-by-trio256", "bonsai-trio200", "bonsai-trio256", 1000,
         False, "bonsai-trio256"),
        ("trio200-holds-small-trio256", "bonsai-trio256", "bonsai-trio200",
         1000, False, "bonsai-trio200"),
        ("trio200-yields-200k", "bonsai-trio256", "bonsai-trio200", 200000,
         False, "bonsai-trio256"),
        ("duo256-not-swallow-trio200", "bonsai-trio200", "bonsai-duo256",
         1000, False, "bonsai-trio200"),
        # Reuse never crosses model families: a compliant slot count is not
        # permission to answer a Qwen request with Bonsai weights.
        ("family-switch-gguf-to-bonsai", "bonsai-quad128", "gguf-duo128", 1000,
         False, "bonsai-quad128"),
        ("family-switch-bonsai-to-gguf", "gguf-range", "bonsai-max", 1000,
         False, "gguf-range"),
        ("family-switch-ninfer-to-gguf", "gguf-range", "balanced", 1000, False,
         "gguf-range"),
        ("family-switch-ninfer-to-bonsai", "bonsai-max", "ultra", 1000, False,
         "bonsai-max"),
    ]:
        check(f"reuse:{label}", E(w, c, e, m), want)

    # Encoder (mmproj) state must be tracked separately from the profile name.
    # It is sticky: once resident it serves text too, so mixed image/text
    # sessions do not reboot the backend on every turn.
    S = nr.same_backend
    for label, cn, cm, p, wm, want in [
        ("exact-text", "balanced", False, "balanced", False, True),
        ("encoder-added-reboots", "gguf-max", False, "gguf-max", True, False),
        ("encoder-kept-reuses", "gguf-max", True, "gguf-max", True, True),
        ("encoder-kept-for-text", "gguf-max", True, "gguf-max", False, True),
        ("bonsai-encoder-kept-for-text", "bonsai-trio256", True,
         "bonsai-trio256", False, True),
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
        ("bonsai-alias-trio", "--alias bonsai-trio256 -c 786432 --parallel 3",
         "bonsai-trio256"),
        ("bonsai-alias-trio200", "--alias bonsai-trio200 -c 614400 --parallel 3",
         "bonsai-trio200"),
        ("bonsai-alias-duo256", "--alias bonsai-duo256 -c 524288 --parallel 2",
         "bonsai-duo256"),
        ("bonsai-alias-hexa", "--alias bonsai-hexa128 -c 786432 --parallel 6",
         "bonsai-hexa128"),
        ("bonsai-alias-penta", "--alias bonsai-penta128 -c 655360 --parallel 5",
         "bonsai-penta128"),
        ("bonsai-no-alias-hexa", "-m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf "
         "--parallel 6 -c 786432", "bonsai-hexa128"),
        ("bonsai-no-alias-penta", "-m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf "
         "--parallel 5 -c 655360", "bonsai-penta128"),
        ("bonsai-no-alias-trio", "-m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf "
         "--parallel 3 -c 786432", "bonsai-trio256"),
        ("bonsai-no-alias-trio200", "-m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf "
         "--parallel 3 -c 614400", "bonsai-trio200"),
        ("bonsai-no-alias-duo256", "-m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf "
         "--parallel 2 -c 524288", "bonsai-duo256"),
        ("bonsai-no-alias-octo",
         "-m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --parallel 8",
         "bonsai-octo64"),
        ("ninfer-131072", "--max-context 131072 --kv-dtype rk8v4", "balanced"),
        ("ninfer-eq-262144", "--max-context=262144", "ultra"),
        ("unknown", "--foo bar", None),
        ("empty", "", None),
    ]:
        check(f"adopt:{label}", I(args), want)
    
    # ---- has_media (directly): every media shape, OpenAI + Anthropic ----
    H = nr.has_media
    for label, d, want in [
        ("img-url", {"messages": [{"content": [img]}]}, True),
        ("input-image", {"messages": [{"content": [{"type": "input_image"}]}]}, True),
        ("input-video", {"messages": [{"content": [{"type": "input_video"}]}]}, True),
        ("video-url", {"messages": [{"content": [{"type": "video_url"}]}]}, True),
        ("image-type", {"messages": [{"content": [{"type": "image"}]}]}, True),
        ("source-image", {"messages": [{"content": [{"source": {"media_type": "image/jpeg"}, "type": "doc"}]}]}, True),
        ("source-audio-false", {"messages": [{"content": [{"source": {"media_type": "audio/mpeg"}}]}]}, False),
        ("source-missing-false", {"messages": [{"content": [{"source": {}}]}]}, False),
        ("plain-string", {"messages": [{"content": "hi"}]}, False),
        ("text-only", {"messages": [{"content": [{"type": "text", "text": "hi"}]}]}, False),
        ("empty-msgs", {"messages": []}, False),
        ("no-messages", {}, False),
        ("non-dict-message", {"messages": ["x"]}, False),
        ("content-not-list", {"messages": [{"content": {"type": "image_url"}}]}, False),
        ("non-dict-part", {"messages": [{"content": ["x", 1, {"type": "text"}]}]}, False),
        ("mixed-text-img", {"messages": [{"content": ["hi", img]}]}, True),
        ("bad-decoded", {"model": "x"}, False),
    ]:
        check(f"media:{label}", H(d), want)
    
    # ---- resolve_effort: client > kwargs > suffix precedence + aliases ----
    R = nr.resolve_effort
    for label, d, s, want in [
        ("client-str", {"reasoning_effort": "medium"}, None, ("medium", "client")),
        ("client-alias", {"reasoning_effort": "minimal"}, None, ("low", "client")),
        ("client-case", {"reasoning_effort": " XHIGH "}, None, ("xhigh", "client")),
        ("client-high", {"reasoning_effort": "high"}, None, ("high", "client")),
        ("client-none", {"reasoning_effort": "none"}, None, ("none", "client")),
        ("kwargs-wins", {"chat_template_kwargs": {"reasoning_effort": "high"}}, "low", ("high", "client-kwargs")),
        ("kwargs-alias", {"chat_template_kwargs": {"reasoning_effort": "minimal"}}, None, ("low", "client-kwargs")),
        ("kwargs-non-dict-falls", {"chat_template_kwargs": "x"}, "low", ("low", "suffix")),
        ("empty-str-falls", {"reasoning_effort": "   "}, "medium", ("medium", "suffix")),
        ("empty-str-no-suffix", {}, None, (None, None)),
        ("non-str-falls", {"reasoning_effort": 1}, "xhigh", ("xhigh", "suffix")),
        ("suffix-only", {}, "low", ("low", "suffix")),
        ("suffix-alias", {}, "minimal", ("low", "suffix")),
        ("suffix-none", {}, "none", ("none", "suffix")),
        ("suffix-raw", {}, "high", ("high", "suffix")),
    ]:
        check(f"effort:{label}", R(d, s), want)
    
    # ---- client_thinking_mode: /v1/messages wins over kwargs ----
    C = nr.client_thinking_mode
    for label, d, path, want in [
        ("msg-enabled", {"thinking": {"type": "enabled"}}, "/v1/messages", "enabled"),
        ("msg-disabled", {"thinking": {"type": "disabled"}}, "/v1/messages", "disabled"),
        ("msg-other-type", {"thinking": {"type": "auto"}}, "/v1/messages", None),
        ("msg-none", {}, "/v1/messages", None),
        ("msg-thinking-str", {"thinking": "enabled"}, "/v1/messages", None),
        ("ctk-true", {"chat_template_kwargs": {"enable_thinking": True}}, "/v1/chat/completions", "enabled"),
        ("ctk-false", {"chat_template_kwargs": {"enable_thinking": False}}, "/v1/chat/completions", "disabled"),
        ("ctk-str-true", {"chat_template_kwargs": {"enable_thinking": "true"}}, "/v1/messages", "enabled"),
        ("ctk-str-false", {"chat_template_kwargs": {"enable_thinking": "false"}}, "/v1/messages", "disabled"),
        ("ctk-str-other", {"chat_template_kwargs": {"enable_thinking": "yes"}}, "/v1/messages", None),
        ("msg-think-wins", {"thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": True}}, "/v1/messages", "disabled"),
        ("chat-kwargs", {"chat_template_kwargs": {"enable_thinking": True}}, "/v1/chat/completions", "enabled"),
        ("extra-not-read", {"extra_params": {"reasoning_budget_tokens": 1}}, "/v1/messages", None),
        ("nothing", {}, "/v1/chat/completions", None),
        ("messages-kwargs-false", {"chat_template_kwargs": {"enable_thinking": False}}, "/v1/messages", "disabled"),
    ]:
        check(f"think:{label}", C(d, path), want)
    
    # ---- has_explicit_budget: body > messages/thinking > extra_params ----
    B = nr.has_explicit_budget
    for label, d, path, want in [
        ("body-key", {"reasoning_budget_tokens": 4096}, "/v1/chat/completions", True),
        ("body-alt-key", {"thinking_budget_tokens": 4096}, "/v1/chat/completions", True),
        ("msg-think-budget", {"thinking": {"type": "enabled", "budget_tokens": 512}}, "/v1/messages", True),
        ("msg-think-disabled", {"thinking": {"type": "disabled"}}, "/v1/messages", False),
        ("msg-think-enabled-nob", {"thinking": {"type": "enabled"}}, "/v1/messages", False),
        ("msg-think-auto-b", {"thinking": {"type": "auto", "budget_tokens": 1}}, "/v1/messages", False),
        ("msg-extra-params-key", {"extra_params": {"reasoning_budget_tokens": 1}}, "/v1/messages", True),
        ("msg-extra-params-other", {"extra_params": {"x": 1}}, "/v1/messages", False),
        ("msg-extra-params-none", {"extra_params": None}, "/v1/messages", False),
        ("nothing", {}, "/v1/chat/completions", False),
        ("chat-extra-params", {"extra_params": {"reasoning_budget_tokens": 1}}, "/v1/chat/completions", False),
    ]:
        check(f"budget:{label}", B(d, path), want)

    # ---- apply_effort_policy: caps for llama.cpp, none for uncapped backends,
    # ---- and a transition message whenever a cap is actually in force ----
    P = nr.apply_effort_policy
    d, meta, notes = {"model": "x"}, {}, []
    P(d, "/v1/chat/completions", "low", True, meta, notes)
    check("effortpol:budget", d.get("reasoning_budget_tokens"), 512)
    check("effortpol:budget-msg", d.get("reasoning_budget_message"), nr.BUDGET_MESSAGE)
    check("effortpol:budget-meta", meta.get("injected_budget"), True)

    for tier in ("low", "medium", "high"):
        d, meta, notes = {"model": "x"}, {}, []
        P(d, "/v1/chat/completions", tier, False, meta, notes)
        check(f"effortpol:no-cap-{tier}", "reasoning_budget_tokens" in d, False)
        check(f"effortpol:no-cap-msg-{tier}", "reasoning_budget_message" in d, False)

    d, meta, notes = {"reasoning_budget_tokens": 999,
                      "reasoning_budget_message": "mine"}, {}, []
    P(d, "/v1/chat/completions", "low", True, meta, notes)
    check("effortpol:client-budget-kept", d.get("reasoning_budget_tokens"), 999)
    check("effortpol:client-budget-no-msg", d.get("reasoning_budget_message"), "mine")

    d, meta, notes = {"reasoning_budget_tokens": 999}, {}, []
    P(d, "/v1/chat/completions", "low", True, meta, notes)
    check("effortpol:client-budget-kept2", d.get("reasoning_budget_tokens"), 999)
    check("effortpol:client-budget-msg", d.get("reasoning_budget_message"), nr.BUDGET_MESSAGE)

    d, meta, notes = {}, {}, []
    P(d, "/v1/chat/completions", "xhigh", True, meta, notes)
    check("effortpol:no-budget", "reasoning_budget_tokens" in d, False)
    check("effortpol:no-budget-msg", "reasoning_budget_message" in d, False)
    
    # ---- route_request: direct coverage of all branches ----
    Q = nr.route_request
    for label, d, want in [
        ("text-plain", {"model": "qwen3.8-27b-128k"}, ("balanced", None, False, [])),
        ("gguf-vision-media", {"model": "qwen3.8-gguf-vision", "messages": [{"content": [img]}]}, ("gguf-vision", None, True, ["media wants mmproj"])),
        ("gguf-range-media", {"model": "qwen3.8-gguf-range", "messages": [{"content": [img]}]}, ("gguf-range", None, True, ["media wants mmproj"])),
        ("vision-media-stays", {"model": "qwen3.6-35b-vision", "messages": [{"content": [img]}]}, ("vision", None, False, [])),
        ("ninfer-media-forces", {"model": "qwen3.8-27b-128k", "messages": [{"content": [img]}]}, ("vision", None, False, ["media forces vision"])),
        ("ninfer-media-none-effort", {"model": "qwen3.8-27b-128k-none", "messages": [{"content": [img]}]}, ("vision", "none", False, ["media forces vision"])),
        ("gguf-vision-no-media", {"model": "qwen3.8-gguf-vision"}, ("gguf-vision", None, True, [])),
        ("bonsai-vision-no-media", {"model": "bonsai-2-27b-vision"}, ("bonsai-vision", None, True, [])),
        ("bonsai-media-lazy", {"model": "bonsai-2-27b-octo64", "messages": [{"content": [img]}]}, ("bonsai-octo64", None, True, ["media wants mmproj"])),
        ("unknown-model", {"model": "foo"}, ("balanced", None, False, [])),
        ("unknown-with-media", {"model": "foo", "messages": [{"content": [img]}]}, ("vision", None, False, ["media forces vision"])),
    ]:
        check(f"route:{label}", Q(d), want)
    
    # ---- message_digest + debug_prompt (prefix-cache forensics, pure) ----
    D = nr.message_digest
    m0 = {"role": "user", "content": "hi"}
    m1 = {"role": "user", "content": "hi", "x": 1}
    check("digest-stable", D(m0), D(m0))
    check("digest-stable2", D(m1), D(m1))
    check("digest-len-8", len(D(m0)), 8)
    check("digest-order", D(m1), D({"role": "user", "content": "hi", "x": 1}))
    check("digest-unicode-order", D({"role": "assistant", "content": "áçë", "tool_calls": []}), D({"tool_calls": [], "role": "assistant", "content": "áçë"}))
    
    # debug_prompt: env-gated forensics (digest log + optional raw body file);
    # must not touch the request path or raise.
    orig_env = dict(os.environ)
    dump_dir = os.path.join("/tmp", "model-router-dump-{}".format(os.getpid()))
    os.makedirs(dump_dir, exist_ok=True)
    try:
        os.environ["MODEL_ROUTER_DIGEST"] = "1"
        os.environ["MODEL_ROUTER_DUMP"] = dump_dir
        nr.debug_prompt({"messages": [m0]}, b'{"model":"x","stream":true}')
        entries = sorted(os.listdir(dump_dir))
        check("dump-writes-body", len(entries) == 1 and entries[0].endswith(".json"), True)
        check("dump-content", open(os.path.join(dump_dir, entries[0]), "rb").read(), b'{"model":"x","stream":true}')
        # No env -> no-op (no extra file).
        os.environ.pop("MODEL_ROUTER_DIGEST", None)
        os.environ.pop("MODEL_ROUTER_DUMP", None)
        before = len(os.listdir(dump_dir))
        nr.debug_prompt({"messages": [m0]}, b'{"model":"y"}')
        check("dump-noop-no-env", len(os.listdir(dump_dir)) == before, True)
    finally:
        os.environ.clear()
        os.environ.update(orig_env)
        import shutil
        shutil.rmtree(dump_dir, ignore_errors=True)
    
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES")
        sys.exit(1)
    print("\nrouter unit tests passed.")


if __name__ == "__main__":
    main()
