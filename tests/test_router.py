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
        "qwen3.8-gguf": ("gguf-range", None),
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
        # Unknown names must not silently pick weights: the general-purpose
        # Qwen tier is the only safe fallback.
        "mystery-model": ("gguf-range", None),
        "mystery-model-low": ("gguf-range", "low"),
    }.items():
        check(f"parse:{name}", nr.parse_model(name), want)

    P = "/v1/chat/completions"
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    TOOLS = {"tools": [{"type": "function",
                        "function": {"name": "x",
                                     "parameters": {"type": "object",
                                                    "properties": {}}}}]}

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
        ("low-is-budgeted", ("qwen3.8-gguf-max-low",),
         ("gguf-max", "low", 1536)),
        ("medium-budget", ("qwen3.8-gguf-max-medium",),
         ("gguf-max", "medium", 3072)),
        ("high-maps-xhigh", ("qwen3.8-gguf-max-high",),
         ("gguf-max", "xhigh", 8192)),
        ("minimal-alias-low", ("qwen3.8-gguf-max-minimal",),
         ("gguf-max", "low", 1536)),
        ("xhigh-unbounded", ("qwen3.8-gguf-max-xhigh",),
         ("gguf-max", "xhigh", "<ABSENT>")),
        ("explicit-wins", ("qwen3.8-gguf-max-low", "hi", "medium"),
         ("gguf-max", "medium", 3072)),
        ("explicit-budget-wins",
         ("qwen3.8-gguf-max-low", "hi", None,
          {"reasoning_budget_tokens": 4096}),
         ("gguf-max", "low", 4096)),
        ("client-thinking-off",
         ("qwen3.8-gguf-max-low", "hi", None,
          {"chat_template_kwargs": {"enable_thinking": False}}),
         ("gguf-max", "<ABSENT>", "<ABSENT>")),
        ("unsupported-dropped", ("qwen3.8-gguf-max", "hi", "max"),
         ("gguf-max", "<ABSENT>", "<ABSENT>")),
        ("unknown-tier-budget", ("mystery-model-low",),
         ("gguf-range", "low", 1536)),
        # The policy follows the RESIDENT family: Bonsai low uses the medium
        # template (the low sentence made it loop) and its own budget.
        ("reuse-budget-follows-tier",
         ("qwen3.8-gguf-max-low", "hi", None, None, "bonsai-max"),
         ("bonsai-max", "medium", 1536)),
        # Media stays on the requested tier now that every backend carries the
        # encoder: no more 35B vision tier.
        ("media-keeps-tier", ("qwen3.8-gguf-range", [img]),
         ("gguf-range", "<ABSENT>", "<ABSENT>")),
        ("plain-untouched", ("qwen3.8-gguf-max",),
         ("gguf-max", "<ABSENT>", "<ABSENT>")),
        ("bonsai-low-budget", ("bonsai-2-27b-quad128-low",),
         ("bonsai-quad128", "medium", 1536)),
        ("bonsai-medium-budget", ("bonsai-2-27b-trio256-medium",),
         ("bonsai-trio256", "medium", 6144)),
        ("bonsai-high-maps-xhigh", ("bonsai-2-27b-octo64-high",),
         ("bonsai-octo64", "xhigh", 8192)),
        # A tool loop raises every Bonsai tier to the agent budget floor; the
        # Qwen tiers keep their tuned caps (they do not show the malformed
        # tool-call failure).
        ("bonsai-low-tools", ("bonsai-2-27b-quad128-low", "hi", None, TOOLS),
         ("bonsai-quad128", "medium", 16384)),
        ("bonsai-medium-tools",
         ("bonsai-2-27b-trio256-medium", "hi", None, TOOLS),
         ("bonsai-trio256", "medium", 16384)),
        ("bonsai-high-tools", ("bonsai-2-27b-octo64-high", "hi", None, TOOLS),
         ("bonsai-octo64", "xhigh", 16384)),
        ("bonsai-xhigh-tools", ("bonsai-2-27b-max-xhigh", "hi", None, TOOLS),
         ("bonsai-max", "xhigh", 16384)),
        ("bonsai-tool-result-only",
         ("bonsai-2-27b-max-low", "hi", None,
          {"messages": [{"role": "tool", "content": "ok"}]}),
         ("bonsai-max", "medium", 16384)),
        ("gguf-low-tools-unchanged",
         ("qwen3.8-gguf-max-low", "hi", None, TOOLS),
         ("gguf-max", "low", 1536)),
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
         ("gguf-max", "low", True, 1536, True)),
        ("gguf-high", ("qwen3.8-gguf-max-high",),
         ("gguf-max", "xhigh", True, 8192, True)),
        ("gguf-none", ("qwen3.8-gguf-max-none",),
         ("gguf-max", "<ABSENT>", False, "<ABSENT>", False)),
        ("explicit-budget",
         ("qwen3.8-gguf-max-low", None,
          {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
         ("gguf-max", "low", True, 4096, False)),
        ("thinking-disabled",
         ("qwen3.8-gguf-max-low", None,
          {"thinking": {"type": "disabled"}}),
         ("gguf-max", "<ABSENT>", "<ABSENT>", "<ABSENT>", False)),
        ("bonsai-low-messages", ("bonsai-2-27b-quad128-low",),
         ("bonsai-quad128", "medium", True, 1536, True)),
        ("bonsai-tool-loop-messages",
         ("bonsai-2-27b-quad128-low", None, TOOLS),
         ("bonsai-quad128", "medium", True, 16384, True)),
        ("reuse-budget-follows-tier",
         ("qwen3.8-gguf-max-low", None, None, "bonsai-max"),
         ("bonsai-max", "medium", True, 1536, True)),
    ]:
        check(f"messages:{label}", messages(*args), want)

    # Bonsai media stays on its own tier with the lazy encoder, like GGUF.
    check("route:bonsai-media-lazy-mmproj",
          nr.route_request({"model": "bonsai-2-27b-octo64",
                            "messages": [{"role": "user", "content": [img]}]}),
          ("bonsai-octo64", None, True, ["media wants mmproj"]))
    check("route:gguf-media-lazy-mmproj",
          nr.route_request({"model": "qwen3.8-gguf-max",
                            "messages": [{"role": "user", "content": [img]}]}),
          ("gguf-max", None, True, ["media wants mmproj"]))
    check("route:text-no-mmproj",
          nr.route_request({"model": "qwen3.8-gguf-max",
                            "messages": [{"role": "user", "content": "hi"}]}),
          ("gguf-max", None, False, []))

    E = nr.effective_profile
    for label, w, c, e, m, want in [
        ("same", "gguf-range", "gguf-range", 1000, False, "gguf-range"),
        ("reuse-smaller", "gguf-max", "gguf-range", 1000, False, "gguf-range"),
        ("overflow-switches", "gguf-max", "gguf-range", 120000, False,
         "gguf-max"),
        ("reuse-bigger", "gguf-range", "gguf-max", 1000, False, "gguf-max"),
        ("nothing-running", "gguf-range", None, 1000, False, "gguf-range"),
        # Media never forces a tier switch: every tier carries mmproj lazily.
        ("media-stays-on-tier", "gguf-max", "gguf-range", 1000, True,
         "gguf-range"),
        ("gguf-vision-fits-stays", "gguf-range", "gguf-vision", 1000, False,
         "gguf-vision"),
        ("gguf-vision-overflow-leaves", "gguf-range", "gguf-vision", 60000,
         False, "gguf-range"),
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
        # Speed-aware reuse: the requested MTP tier (97 tok/s) beats the
        # resident base (33) by 2.9x, so switch; the reverse keeps the faster
        # resident. Same-group switches have a lower bar (the slot restores the
        # KV) than base <-> MTP (which re-prefills the context).
        ("speed-switch-mtp", "gguf-single200", "gguf-max", 1000, False,
         "gguf-single200"),
        ("speed-keep-faster-resident", "gguf-max", "gguf-single200", 1000,
         False, "gguf-single200"),
        ("speed-switch-mtp-from-duo", "gguf-single200", "gguf-duo128", 1000,
         False, "gguf-single200"),
        ("speed-switch-speed-from-range", "gguf-speed", "gguf-range", 1000,
         False, "gguf-speed"),
        ("speed-keep-equal-base", "gguf-max", "gguf-range", 1000, False,
         "gguf-range"),
        ("speed-bonsai-max-keeps-duo", "bonsai-max", "bonsai-duo256", 1000,
         False, "bonsai-duo256"),
        ("speed-bonsai-max-keeps-trio", "bonsai-max", "bonsai-trio200", 1000,
         False, "bonsai-trio200"),
        ("speed-cross-family-switches", "gguf-single200", "bonsai-duo256",
         1000, False, "gguf-single200"),
        ("speed-vision-same", "gguf-vision", "gguf-range", 1000, False,
         "gguf-range"),
    ]:
        check(f"reuse:{label}", E(w, c, e, m), want)

    # Encoder (mmproj) state must be tracked separately from the profile name.
    # It is sticky: once resident it serves text too, so mixed image/text
    # sessions do not reboot the backend on every turn.
    S = nr.same_backend
    for label, cn, cm, p, wm, want in [
        ("exact-text", "gguf-range", False, "gguf-range", False, True),
        ("encoder-added-reboots", "gguf-max", False, "gguf-max", True, False),
        ("encoder-kept-reuses", "gguf-max", True, "gguf-max", True, True),
        ("encoder-kept-for-text", "gguf-max", True, "gguf-max", False, True),
        ("bonsai-encoder-kept-for-text", "bonsai-trio256", True,
         "bonsai-trio256", False, True),
        ("tier-differs", "gguf-range", False, "gguf-max", False, False),
        ("nothing-running", None, False, "gguf-range", False, False),
    ]:
        check(f"same:{label}", S(cn, cm, p, wm), want)

    # Adoption must recognize a live container from its server args, else the
    # router reboots a healthy backend on startup.
    I = nr.infer_profile
    for label, args, want in [
        ("llama-alias-single200",
         "-m /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --alias "
         "qwen3.8-gguf-single200 -c 204800 --parallel 1 --mmproj /m/mmproj.gguf",
         "gguf-single200"),
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

    # ---- apply_effort_policy: per-family safety nets, and a transition
    # ---- message whenever a cap is actually in force ----
    P = nr.apply_effort_policy
    for fam, want in (("gguf", {"low": 1536, "medium": 3072, "high": 8192,
                                "xhigh": None}),
                      ("bonsai", {"low": 1536, "medium": 6144, "high": 8192,
                                  "xhigh": None})):
        for tier, budget in want.items():
            d, meta, notes = {"model": "x"}, {}, []
            P(d, "/v1/chat/completions", tier, fam, meta, notes)
            check(f"effortpol:{fam}-{tier}-budget",
                  d.get("reasoning_budget_tokens"), budget)
            check(f"effortpol:{fam}-{tier}-msg",
                  "reasoning_budget_message" in d, budget is not None)
            check(f"effortpol:{fam}-{tier}-meta",
                  bool(meta.get("injected_budget")), budget is not None)

    # An unknown family (no launcher matched) still gets the gguf safety net:
    # running uncapped by accident is worse than a generous cap.
    d, meta, notes = {"model": "x"}, {}, []
    P(d, "/v1/chat/completions", "low", None, meta, notes)
    check("effortpol:unknown-family-cap",
          d.get("reasoning_budget_tokens"), 1536)

    # Tool loops never get the forced transition message: the 16k floor means
    # the natural reasoning fits, and the message itself anchored the model
    # into emitting a malformed call after it.
    for tier in ("low", "medium", "high", "xhigh"):
        d = {"model": "x", **TOOLS}
        meta, notes = {}, []
        P(d, "/v1/chat/completions", tier, "bonsai", meta, notes)
        check(f"effortpol:tools-{tier}-floor",
              d.get("reasoning_budget_tokens"), 16384)
        check(f"effortpol:tools-{tier}-no-msg",
              "reasoning_budget_message" in d, False)
        check(f"effortpol:tools-{tier}-meta",
              bool(meta.get("tool_loop")), True)
    # A small client max_tokens still bounds it (answer reserve).
    d = {"model": "x", "max_tokens": 4096, **TOOLS}
    meta, notes = {}, []
    P(d, "/v1/chat/completions", "low", "bonsai", meta, notes)
    check("effortpol:tools-dynamic-cap",
          d.get("reasoning_budget_tokens"), 4096 - nr.ANSWER_RESERVE)
    check("effortpol:tools-dynamic-cap-no-msg",
          "reasoning_budget_message" in d, False)
    # An explicit client budget is never overwritten.
    d = {"model": "x", "reasoning_budget_tokens": 2048, **TOOLS}
    meta, notes = {}, []
    P(d, "/v1/chat/completions", "low", "bonsai", meta, notes)
    check("effortpol:tools-client-budget",
          d.get("reasoning_budget_tokens"), 2048)
    check("effortpol:tools-client-budget-no-msg",
          "reasoning_budget_message" in d, False)

    # ---- style injection + dynamic budget ----
    def body_for(model, extra=None, path="/v1/chat/completions"):
        d = {"model": model,
             "messages": [{"role": "system", "content": "CLIENT SYSTEM"},
                          {"role": "user", "content": "hi"}]}
        if extra:
            d.update(extra)
        _p, raw, _n, _m, _meta = nr.normalize_chat_body(
            path, json.dumps(d).encode())
        return json.loads(raw)

    def system_of(out):
        for m in out.get("messages") or []:
            if m.get("role") == "system":
                return m.get("content") or ""
        return ""

    lo = body_for("qwen3.8-gguf-max-low")
    check("style:gguf-low-fewshot", nr.FEWSHOT_STYLE in system_of(lo), True)
    check("style:gguf-low-client-kept", "CLIENT SYSTEM" in system_of(lo), True)
    check("style:gguf-low-template", lo.get("reasoning_effort"), "low")
    md = body_for("qwen3.8-gguf-max-medium")
    check("style:gguf-medium-fewshot",
          nr.FEWSHOT_STYLE in system_of(md), True)
    hi = body_for("qwen3.8-gguf-max-high")
    check("style:gguf-high-none",
          nr.FEWSHOT_STYLE not in system_of(hi), True)
    bo = body_for("bonsai-2-27b-duo256-low")
    check("style:bonsai-low-none", nr.FEWSHOT_STYLE not in system_of(bo), True)
    check("style:bonsai-low-template", bo.get("reasoning_effort"), "medium")
    no = body_for("qwen3.8-gguf-max-none")
    check("style:none-none", nr.FEWSHOT_STYLE not in system_of(no), True)
    off = body_for("qwen3.8-gguf-max-low",
                   {"chat_template_kwargs": {"enable_thinking": False}})
    check("style:client-off-none",
          nr.FEWSHOT_STYLE not in system_of(off), True)

    def budget_for(model, extra):
        return body_for(model, extra).get("reasoning_budget_tokens")

    check("dyn:low-4k", budget_for("qwen3.8-gguf-max-low",
                                   {"max_tokens": 4096}), 1536)
    check("dyn:medium-4k", budget_for("qwen3.8-gguf-max-medium",
                                      {"max_tokens": 4096}), 2560)
    check("dyn:xhigh-4k", budget_for("qwen3.8-gguf-max-xhigh",
                                     {"max_tokens": 4096}), 2560)
    check("dyn:xhigh-8k", budget_for("qwen3.8-gguf-max-xhigh",
                                     {"max_tokens": 8192}), 6656)
    check("dyn:xhigh-none", budget_for("qwen3.8-gguf-max-xhigh", {}), None)
    check("dyn:low-8k", budget_for("qwen3.8-gguf-max-low",
                                   {"max_tokens": 8192}), 1536)
    check("dyn:bonsai-low-4k", budget_for("bonsai-2-27b-duo256-low",
                                          {"max_tokens": 4096}), 1536)
    check("dyn:max-completion", budget_for("qwen3.8-gguf-max-xhigh",
                                           {"max_completion_tokens": 4096}), 2560)
    check("dyn:low-1k", budget_for("qwen3.8-gguf-max-low",
                                   {"max_tokens": 1024}), 512)
    check("dyn:low-tiny", budget_for("qwen3.8-gguf-max-low",
                                     {"max_tokens": 512}), 256)

    mlo_body = {"model": "qwen3.8-gguf-max-low", "max_tokens": 4096,
                "system": "CLIENT SYSTEM",
                "messages": [{"role": "user", "content": "hi"}]}
    _p, raw, _n, _m, _meta = nr.normalize_chat_body(
        "/v1/messages", json.dumps(mlo_body).encode())
    mlo = json.loads(raw)
    check("style:messages-low-fewshot",
          nr.FEWSHOT_STYLE in (mlo.get("system") or ""), True)
    check("style:messages-client-kept",
          "CLIENT SYSTEM" in (mlo.get("system") or ""), True)
    check("dyn:messages-low-4k",
          (mlo.get("thinking") or {}).get("budget_tokens"), 1536)

    d, meta, notes = {"reasoning_budget_tokens": 999,
                      "reasoning_budget_message": "mine"}, {}, []
    P(d, "/v1/chat/completions", "low", "gguf", meta, notes)
    check("effortpol:client-budget-kept", d.get("reasoning_budget_tokens"), 999)
    check("effortpol:client-budget-no-msg", d.get("reasoning_budget_message"), "mine")

    d, meta, notes = {"reasoning_budget_tokens": 999}, {}, []
    P(d, "/v1/chat/completions", "low", "gguf", meta, notes)
    check("effortpol:client-budget-kept2", d.get("reasoning_budget_tokens"), 999)
    check("effortpol:client-budget-msg", d.get("reasoning_budget_message"), nr.BUDGET_MESSAGE)

    # A cap in force on a client budget must never be reported as absent.
    d, meta, notes = {"reasoning_budget_tokens": 999}, {}, []
    P(d, "/v1/chat/completions", "low", "bonsai", meta, notes)
    check("effortpol:client-budget-note",
          any("kept from client" in n for n in notes), True)
    
    # ---- route_request: direct coverage of all branches ----
    Q = nr.route_request
    for label, d, want in [
        ("text-plain", {"model": "qwen3.8-gguf-range"}, ("gguf-range", None, False, [])),
        ("gguf-vision-media", {"model": "qwen3.8-gguf-vision", "messages": [{"content": [img]}]}, ("gguf-vision", None, True, ["media wants mmproj"])),
        ("gguf-range-media", {"model": "qwen3.8-gguf-range", "messages": [{"content": [img]}]}, ("gguf-range", None, True, ["media wants mmproj"])),
        ("vision-media-lazy", {"model": "qwen3.8-gguf-vision", "messages": [{"content": [img]}]}, ("gguf-vision", None, True, ["media wants mmproj"])),
        ("gguf-vision-no-media", {"model": "qwen3.8-gguf-vision"}, ("gguf-vision", None, True, [])),
        ("bonsai-vision-no-media", {"model": "bonsai-2-27b-vision"}, ("bonsai-vision", None, True, [])),
        ("bonsai-media-lazy", {"model": "bonsai-2-27b-octo64", "messages": [{"content": [img]}]}, ("bonsai-octo64", None, True, ["media wants mmproj"])),
        ("unknown-model", {"model": "foo"}, ("gguf-range", None, False, [])),
        ("unknown-with-media", {"model": "foo", "messages": [{"content": [img]}]}, ("gguf-range", None, True, ["media wants mmproj"])),
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

    # ---- prefill gate prefix predictor: only the final message may differ
    # between a hot continuation and a cold prompt. ----
    p1 = {"model": "x", "stream": True, "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]}
    p2 = dict(p1)
    p2["messages"] = p1["messages"] + [
        {"role": "user", "content": "different next"}]
    p3 = dict(p2)
    p3["messages"] = p1["messages"] + [
        {"role": "user", "content": "another next"}]
    check("gate-prefix-same", nr.prompt_prefix_key(p2),
          nr.prompt_prefix_key(p3))
    check("gate-prefix-empty", nr.prompt_prefix_key({"messages": []}), None)
    nr.recent_prefixes.clear()
    check("gate-cold-before-admit", nr.prompt_is_hot(p1), False)
    nr.remember_prompt_prefix(p1)
    check("gate-hot-after-admit", nr.prompt_is_hot(p2), True)

    # ---- gate policy: pure decision, no backend/docker needed ----
    check("gate:pass-idle", nr.gate_decision(False, False, True), None)
    check("gate:pass-warm-big", nr.gate_decision(False, True, False), None)
    check("gate:pass-small-cold", nr.gate_decision(False, True, False), None)
    check("gate:hold-prefill", nr.gate_decision(True, False, False),
          "prefill")
    check("gate:hold-prefill-wins", nr.gate_decision(True, True, True),
          "prefill")
    check("gate:hold-cold-vs-decode", nr.gate_decision(False, True, True),
          "cold-vs-decode")

    # ---- /slots phase classification: samples measured on trio200 ----
    # Decoding: cache 87.873, 213 processed, 154 tokens sampled.
    live = {"id": 2, "is_processing": True, "n_prompt_tokens": 88240,
            "n_prompt_tokens_cache": 87873, "n_prompt_tokens_processed": 213,
            "next_token": [{"has_next_token": True, "n_decoded": 154}]}
    check("slots:next-token", nr.next_token_info(live), (154, True))
    check("slots:no-next-token", nr.next_token_info({"next_token": []}),
          (0, False))
    check("slots:bad-next-token", nr.next_token_info({"next_token": ["x"]}),
          (0, False))
    dec = nr.slot_phase(live)
    check("slots:decode-generating", dec["generating"], True)
    check("slots:decode-not-prefilling", dec["prefilling"], False)
    check("slots:decode-prompt-len", dec["prompt_len"], 88086)

    # Prefilling: the whole 10.1k prompt reported as 2048/4096/... because
    # n_prompt_tokens only covers KV cache + the queued batch, and no token
    # has been sampled yet. This is the sample the old `rest > 2048` rule
    # misread as idle.
    pre = {"id": 1, "is_processing": True, "n_prompt_tokens": 4096,
           "n_prompt_tokens_cache": 0, "n_prompt_tokens_processed": 2048,
           "next_token": [{"has_next_token": False, "n_decoded": 0}]}
    p = nr.slot_phase(pre)
    check("slots:prefill-detected", p["prefilling"], True)
    check("slots:prefill-not-generating", p["generating"], False)
    check("slots:prefill-in-flight", p["in_flight"], 4096)

    # Warm continuation: most of the prompt is cached, so the slot is only
    # working on a handful of new tokens -> not a prefill worth gating on.
    warm = {"id": 0, "is_processing": True, "n_prompt_tokens": 87400,
            "n_prompt_tokens_cache": 87200, "n_prompt_tokens_processed": 0,
            "next_token": [{"has_next_token": False, "n_decoded": 0}]}
    check("slots:warm-short-not-prefill", nr.slot_phase(warm)["prefilling"],
          False)
    idle = nr.slot_phase({"id": 3, "is_processing": False})
    check("slots:idle-not-prefill", (idle["prefilling"],
                                     idle["generating"]), (False, False))
    check("slots:not-a-dict", nr.slot_phase("x"), None)

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

    # ---- slot persistence helpers: the dump belongs to a weights file, not
    # to a tier or a slot count, and the restore must never wrap a dump into
    # the wrong slot. ----
    check("slot:same-weights-same-key",
          nr.slot_weights_key("gguf-range") == nr.slot_weights_key("gguf-max"),
          True)
    check("slot:different-file-different-key",
          nr.slot_weights_key("gguf-speed") != nr.slot_weights_key("gguf-range"),
          True)
    check("slot:families-differ",
          nr.slot_weights_key("gguf-range") != nr.slot_weights_key("bonsai-trio200"),
          True)
    name0 = nr.slot_file_name("gguf-duo", 0)
    name1 = nr.slot_file_name("gguf-duo", 1)
    name3 = nr.slot_file_name("gguf-quad", 3)
    check("slot:filename-shaped",
          name0.endswith("-slot0.bin") and name0.startswith("gguf-"), True)
    check("slot:filename-safe",
          "/" not in name0 and ".." not in name0 and len(name0) < 255, True)
    names = [name0, name1, name0, name3, "unrelated.bin"]
    check("slot:restore-duo",
          nr.slot_restore_candidates("gguf-duo", names),
          [(0, name0), (1, name1)])
    check("slot:restore-range",
          nr.slot_restore_candidates("gguf-range", names),
          [(0, name0)])
    # A 4-slot tier accepts the duo dump in slots 0/1 and its own slot 3.
    check("slot:restore-quad",
          nr.slot_restore_candidates("gguf-quad", names),
          [(0, name0), (1, name1), (3, name3)])
    check("slot:restore-empty", nr.slot_restore_candidates("gguf-duo", []), [])
    phases = {"slots": [
        {"id": 2, "proc": False, "prompt_len": 5000},
        {"id": 0, "proc": True, "prompt_len": 9000},
        {"id": 1, "proc": False, "prompt_len": 100},
    ]}
    check("slot:to-save", nr.slots_to_save(phases), [2])
    check("slot:to-save-none", nr.slots_to_save(None), [])

    # ---- table invariants: a profile missing from one table silently changes
    # routing (e.g. no window -> never reused, no VISION_CAPABLE -> media
    # reboots a whole tier). Keep the tables in lockstep. ----
    check("inv:profiles-window", nr.PROFILES - set(nr.PROFILE_WINDOWS), set())
    check("inv:profiles-slots", nr.PROFILES - set(nr.PROFILE_SLOTS), set())
    check("inv:profiles-family",
          {p for p in nr.PROFILES if nr.family(p) is None}, set())
    check("inv:profiles-vision", nr.PROFILES - nr.VISION_CAPABLE, set())
    check("inv:gguf-files", set(nr.GGUF_FILES) - nr.PROFILES, set())
    check("inv:bonsai-files", set(nr.BONSAI_FILES) - nr.PROFILES, set())
    check("inv:profile-speed", set(nr.PROFILE_SPEED), nr.PROFILES)
    check("inv:profile-group", set(nr.PROFILE_GROUP), nr.PROFILES)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES")
        sys.exit(1)
    print("\nrouter unit tests passed.")


if __name__ == "__main__":
    main()
