#!/usr/bin/env python3
"""A/B the reasoning-effort mechanisms on a live llama-server.

The current policy caps the thinking tokens and leans on a single template
sentence for `low`; the reference measurements show the cap truncates instead
of making the model reason less. This probe compares mechanisms that change
the model's behaviour before the cap:

  off_direct / off_plan   thinking disabled, answer or a one-line plan
  no_think_system         Qwen `/no_think` soft switch in the system message
  strong_system           a stronger brevity instruction than the template's
  fewshot                 one concise-reasoning example
  two_pass                no-think plan, then no-think answer
  sampling                strong_system + lower temperature / presence penalty
  budget controls         low at 2048 vs 3072, medium at 3072 vs 2048

It measures reasoning tokens (via /tokenize of message.reasoning_content),
answer tokens, wall time, finish_reason, budget-cap hits and accuracy on a
small verifiable set. Talk to the BACKEND (default :8080), not the router: the
router would apply the very policy under test.

Usage:
  scripts/effort-probe.py --suite pilot
  scripts/effort-probe.py --suite full --candidates xhigh current_low ...
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request

BUDGET_MSG = "\n\nOkay, I have thought enough. Time to write the final answer."
ANSWER_TAIL = " End with ANSWER: <value>."
NUM = re.compile(r"-?\d+(?:\.\d+)?")
ANS = re.compile(r"ANSWER:\s*(.+)", re.I)

# Production style texts (must match model-router.py).
FEWSHOT_STYLE = (
    "Example of the desired style:\n"
    "Question: What is 17*24?\n"
    "17*24 = 17*20 + 17*4 = 408.\n"
    "408\n\n"
    "Solve the next task in the same concise style: at most one sentence of "
    "reasoning, then the answer on its own line, without labels."
)
BRIEF_STYLE = (
    "Think briefly: at most one short paragraph. Then answer. Do not "
    "enumerate alternatives."
)
CLIENT_SYSTEM = ""

PROMPTS = {
    # -- graded: exact answers ------------------------------------------------
    "math_wh": {
        "cls": "difficult", "expected": "604",
        "text": "A warehouse starts with 1,200 units. On Monday it ships 37% "
                "of the stock. On Tuesday it receives 150 units. On Wednesday "
                "it ships one third of the remaining units. How many units "
                "remain?" + ANSWER_TAIL,
    },
    "logic_boxes": {
        "cls": "difficult", "expected": "apples",
        "text": "Three boxes are labeled 'apples', 'oranges' and 'both'. "
                "Every label is wrong. You may draw exactly one fruit from one "
                "box. You draw from the box labeled 'both' and get an apple. "
                "Which box contains only oranges?" + ANSWER_TAIL,
    },
    "math_train": {
        "cls": "difficult", "expected": "70",
        "text": "A train travels 240 km at 80 km/h, then 180 km at 60 km/h. "
                "What is the average speed for the whole trip in km/h?" +
                ANSWER_TAIL,
    },
    "math_work": {
        "cls": "difficult", "expected": "12",
        "text": "A job takes 6 workers 12 days. After 4 days, 2 workers leave. "
                "How many more days do the remaining workers need?" + ANSWER_TAIL,
    },
    "math_discount": {
        "cls": "trivial", "expected": "100",
        "text": "A shirt costs $80 after a 20% discount. What was the original "
                "price in dollars?" + ANSWER_TAIL,
    },
    "math_pct": {
        "cls": "trivial", "expected": "36",
        "text": "What is 15% of 240?" + ANSWER_TAIL,
    },
    "math_cube": {
        "cls": "trivial", "expected": "64",
        "text": "A cube has edge length 4 cm. What is its volume in cm^3?" +
                ANSWER_TAIL,
    },
    "needle": {
        "cls": "needle", "expected": "ZX-4471",
        "text": "Read the log below. The access code is ZX-4471. The code is "
                "also the only token that starts with ZX in the log. "
                + " ".join(f"entry{i:04d} status=ok latency={i % 97}ms" for i in range(900))
                + " End of log. What is the access code?" + ANSWER_TAIL,
    },
    "realistic_pool": {
        "cls": "realistic", "expected": None,
        "text": "We run a 27B model behind a single llama-server with three "
                "slots. Under load, a long tool-call prompt prefills for about "
                "60 seconds and the other slots decode at roughly 1 token/s. "
                "Propose a concrete mitigation and the trade-offs." + ANSWER_TAIL,
    },
    "code_task": {
        "cls": "realistic", "expected": None,
        "text": "Write a Python function add(a, b) that returns the sum. "
                "Reply with only the function code, no labels or prose.",
    },
}

PILOT = ["math_wh", "logic_boxes", "realistic_pool"]
FULL = list(PROMPTS)


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ntok(base, text):
    if not text:
        return 0
    try:
        d = post(base + "/tokenize", {"content": text}, timeout=30)
        return len(d.get("tokens") or [])
    except Exception:
        return max(1, len(text) // 4)


def user(text):
    return {"role": "user", "content": text}


def with_system(system, text):
    return [{"role": "system", "content": system}, user(text)]


def base_body(model, messages, effort=None, budget=None, ctk=None,
              max_tokens=4096, temperature=0.0, seed=42, presence=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False, "seed": seed,
            "reasoning_format": "deepseek"}
    if effort:
        body["reasoning_effort"] = effort
    if budget:
        body["reasoning_budget_tokens"] = budget
        body["reasoning_budget_message"] = BUDGET_MSG
    if ctk:
        body["chat_template_kwargs"] = ctk
    if presence is not None:
        body["presence_penalty"] = presence
    return body


def grade(expected, content):
    if expected is None:
        return None
    m = ANS.findall(content or "")
    ans = m[-1].strip() if m else (content or "").strip().splitlines()[-1] if content.strip() else ""
    if re.fullmatch(r"-?\d+(?:\.\d+)?", expected):
        nums = NUM.findall(ans.replace(",", ""))
        return bool(nums) and abs(float(nums[-1]) - float(expected)) < 1e-6
    return expected.lower() in ans.lower()


class Runner:
    def __init__(self, base, model, seed):
        self.base = base
        self.model = model
        self.seed = seed

    def one(self, prompt, label, body, plan=None):
        t0 = time.monotonic()
        try:
            d = post(self.base + "/v1/chat/completions", body)
            err = None
        except Exception as e:
            d, err = None, repr(e)
        wall = time.monotonic() - t0
        if d is None:
            return {"label": label, "prompt": prompt, "error": err,
                    "wall_s": round(wall, 1)}
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        rc = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        return {
            "label": label, "prompt": prompt,
            "reasoning_tokens": ntok(self.base, rc),
            "answer_tokens": ntok(self.base, content),
            "plan_tokens": ntok(self.base, plan) if plan else 0,
            "wall_s": round(wall, 1),
            "finish": ch.get("finish_reason"),
            "accuracy": grade(PROMPTS[prompt]["expected"], content),
            "reasoning_snip": rc[:160].replace("\n", " "),
            "content_snip": content[:160].replace("\n", " "),
            "usage": d.get("usage") or {},
        }

    def two_pass(self, prompt):
        p = PROMPTS[prompt]["text"]
        plan_body = base_body(
            self.model,
            with_system("Write one sentence of plan only. Do not solve the problem.", p),
            ctk={"enable_thinking": False}, max_tokens=80, temperature=0.0,
            seed=self.seed)
        t0 = time.monotonic()
        try:
            d1 = post(self.base + "/v1/chat/completions", plan_body)
            plan = ((d1.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        except Exception as e:
            return {"label": "two_pass", "prompt": prompt, "error": repr(e),
                    "wall_s": round(time.monotonic() - t0, 1)}
        ans_body = base_body(
            self.model,
            with_system("Answer directly using the plan. Do not reason. End with "
                        "the requested ANSWER line.",
                        f"Plan: {plan.strip()}\n\nProblem: {p}"),
            ctk={"enable_thinking": False}, max_tokens=2048, temperature=0.0,
            seed=self.seed)
        t1 = time.monotonic()
        try:
            d2 = post(self.base + "/v1/chat/completions", ans_body)
            err = None
        except Exception as e:
            d2, err = None, repr(e)
        wall = (t1 - t0) + (time.monotonic() - t1)
        if d2 is None:
            return {"label": "two_pass", "prompt": prompt, "error": err,
                    "wall_s": round(wall, 1)}
        ch = (d2.get("choices") or [{}])[0]
        content = (ch.get("message") or {}).get("content") or ""
        return {
            "label": "two_pass", "prompt": prompt,
            "reasoning_tokens": 0,
            "answer_tokens": ntok(self.base, content),
            "plan_tokens": ntok(self.base, plan),
            "wall_s": round(wall, 1),
            "finish": ch.get("finish_reason"),
            "accuracy": grade(PROMPTS[prompt]["expected"], content),
            "reasoning_snip": "",
            "content_snip": content[:160].replace("\n", " "),
            "plan_snip": plan.strip()[:120].replace("\n", " "),
        }


def candidates(runner):
    m = runner.model
    off = {"enable_thinking": False}
    return {
        # -- reference points -------------------------------------------------
        "xhigh": lambda p: runner.one(p, "xhigh", base_body(
            m, [user(PROMPTS[p]["text"])], effort="xhigh", seed=runner.seed)),
        "current_low": lambda p: runner.one(p, "current_low", base_body(
            m, [user(PROMPTS[p]["text"])], effort="low", budget=2048,
            seed=runner.seed)),
        "current_medium": lambda p: runner.one(p, "current_medium", base_body(
            m, [user(PROMPTS[p]["text"])], effort="medium", budget=3072,
            seed=runner.seed)),
        "low_3072": lambda p: runner.one(p, "low_3072", base_body(
            m, [user(PROMPTS[p]["text"])], effort="low", budget=3072,
            seed=runner.seed)),
        # -- alternatives -----------------------------------------------------
        "off_direct": lambda p: runner.one(p, "off_direct", base_body(
            m, with_system("Answer directly. Do not produce reasoning or "
                           "preamble. Keep the final answer terse.", PROMPTS[p]["text"]),
            ctk=off, seed=runner.seed)),
        "off_plan": lambda p: runner.one(p, "off_plan", base_body(
            m, with_system("Write one short sentence of plan, then the answer. "
                           "Do not produce any other reasoning.", PROMPTS[p]["text"]),
            ctk=off, seed=runner.seed)),
        "no_think_system": lambda p: runner.one(p, "no_think_system", base_body(
            m, with_system("/no_think", PROMPTS[p]["text"]), effort="low",
            budget=2048, seed=runner.seed)),
        "strong_system": lambda p: runner.one(p, "strong_system", base_body(
            m, with_system("Do not use chain-of-thought. If the answer is not "
                           "immediate, write at most one sentence of reasoning, "
                           "then answer. Do not enumerate alternatives or "
                           "validate them. End with the requested ANSWER line.",
                           PROMPTS[p]["text"]), effort="low", budget=2048,
            seed=runner.seed)),
        "fewshot": lambda p: runner.one(p, "fewshot", base_body(
            m, with_system(
                "Example of the desired style:\n"
                "Q: What is 17*24? End with ANSWER: <number>.\n"
                "Reasoning: 17*24 = 17*20 + 17*4 = 408.\n"
                "ANSWER: 408\n\n"
                "Now solve the next problem in the same concise style. At most "
                "one sentence of reasoning.", PROMPTS[p]["text"]), effort="low",
            budget=2048, seed=runner.seed)),
        "two_pass": lambda p: runner.two_pass(p),
        "sampling": lambda p: runner.one(p, "sampling", base_body(
            m, with_system("Do not use chain-of-thought. If the answer is not "
                           "immediate, write at most one sentence of reasoning, "
                           "then answer. End with the requested ANSWER line.",
                           PROMPTS[p]["text"]), effort="low", budget=2048,
            temperature=0.7, seed=runner.seed, presence=0.1)),
        "medium_brief": lambda p: runner.one(p, "medium_brief", base_body(
            m, with_system("Think briefly: at most one short paragraph. Then "
                           "answer. End with the requested ANSWER line.",
                           PROMPTS[p]["text"]), effort="medium", budget=2048,
            seed=runner.seed)),
        # -- Bonsai family (its natural tail is 730-4242; budgets above it) ---
        "bonsai_low": lambda p: runner.one(p, "bonsai_low", base_body(
            m, [user(PROMPTS[p]["text"])], effort="low", budget=4608,
            seed=runner.seed)),
        "bonsai_medium": lambda p: runner.one(p, "bonsai_medium", base_body(
            m, [user(PROMPTS[p]["text"])], effort="medium", budget=6144,
            seed=runner.seed)),
        "bonsai_fewshot": lambda p: runner.one(p, "bonsai_fewshot", base_body(
            m, with_system(
                "Example of the desired style:\n"
                "Q: What is 17*24? End with ANSWER: <number>.\n"
                "Reasoning: 17*24 = 17*20 + 17*4 = 408.\n"
                "ANSWER: 408\n\n"
                "Now solve the next problem in the same concise style. At most "
                "one sentence of reasoning.", PROMPTS[p]["text"]), effort="low",
            budget=4608, seed=runner.seed)),
        "bonsai_brief": lambda p: runner.one(p, "bonsai_brief", base_body(
            m, with_system("Think briefly: at most one short paragraph. Then "
                           "answer. End with the requested ANSWER line.",
                           PROMPTS[p]["text"]), effort="medium", budget=6144,
            seed=runner.seed)),
        "bonsai_xhigh": lambda p: runner.one(p, "bonsai_xhigh", base_body(
            m, [user(PROMPTS[p]["text"])], effort="xhigh", seed=runner.seed)),
        # -- dynamic reasoning budget = max_tokens - answer reserve ------------
        "bonsai_low_dyn": lambda p: runner.one(p, "bonsai_low_dyn", base_body(
            m, [user(PROMPTS[p]["text"])], effort="low", budget=2560,
            seed=runner.seed)),
        "bonsai_fewshot_dyn": lambda p: runner.one(p, "bonsai_fewshot_dyn", base_body(
            m, with_system(
                "Example of the desired style:\n"
                "Q: What is 17*24? End with ANSWER: <number>.\n"
                "Reasoning: 17*24 = 17*20 + 17*4 = 408.\n"
                "ANSWER: 408\n\n"
                "Now solve the next problem in the same concise style. At most "
                "one sentence of reasoning.", PROMPTS[p]["text"]), effort="low",
            budget=2560, seed=runner.seed)),
        "bonsai_medium_dyn": lambda p: runner.one(p, "bonsai_medium_dyn", base_body(
            m, [user(PROMPTS[p]["text"])], effort="medium", budget=2560,
            seed=runner.seed)),
        "bonsai_brief_dyn": lambda p: runner.one(p, "bonsai_brief_dyn", base_body(
            m, with_system("Think briefly: at most one short paragraph. Then "
                           "answer. End with the requested ANSWER line.",
                           PROMPTS[p]["text"]), effort="medium", budget=2560,
            seed=runner.seed)),
        "bonsai_xhigh_dyn": lambda p: runner.one(p, "bonsai_xhigh_dyn", base_body(
            m, [user(PROMPTS[p]["text"])], effort="xhigh", budget=2560,
            seed=runner.seed)),
        "bonsai_fewshot_medium_dyn": lambda p: runner.one(
            p, "bonsai_fewshot_medium_dyn", base_body(
                m, with_system(
                    "Example of the desired style:\n"
                    "Q: What is 17*24? End with ANSWER: <number>.\n"
                    "Reasoning: 17*24 = 17*20 + 17*4 = 408.\n"
                    "ANSWER: 408\n\n"
                    "Now solve the next problem in the same concise style. At most "
                    "one sentence of reasoning.", PROMPTS[p]["text"]),
                effort="medium", budget=2560, seed=runner.seed)),
        "qwen_xhigh_dyn": lambda p: runner.one(p, "qwen_xhigh_dyn", base_body(
            m, [user(PROMPTS[p]["text"])], effort="xhigh", budget=2560,
            seed=runner.seed)),
        # -- real agent system prompt: the few-shot must survive a long system;
        # -- prepend vs append tests recency. -------------------------------
        "real_qwen_old_low": lambda p: runner.one(p, "real_qwen_old_low", base_body(
            m, with_system(CLIENT_SYSTEM, PROMPTS[p]["text"]),
            effort="medium", budget=2048, seed=runner.seed)),
        "real_qwen_low_prepend": lambda p: runner.one(
            p, "real_qwen_low_prepend", base_body(
                m, with_system(FEWSHOT_STYLE + "\n\n" + CLIENT_SYSTEM,
                               PROMPTS[p]["text"]),
                effort="low", budget=1536, seed=runner.seed)),
        "real_qwen_medium_prepend": lambda p: runner.one(
            p, "real_qwen_medium_prepend", base_body(
                m, with_system(FEWSHOT_STYLE + "\n\n" + CLIENT_SYSTEM,
                               PROMPTS[p]["text"]),
                effort="medium", budget=2560, seed=runner.seed)),
        "real_qwen_low_append": lambda p: runner.one(
            p, "real_qwen_low_append", base_body(
                m, with_system(CLIENT_SYSTEM + "\n\n" + FEWSHOT_STYLE,
                               PROMPTS[p]["text"]),
                effort="low", budget=2560, seed=runner.seed)),
        "real_qwen_medium_plain": lambda p: runner.one(
            p, "real_qwen_medium_plain", base_body(
                m, with_system(CLIENT_SYSTEM, PROMPTS[p]["text"]),
                effort="medium", budget=2048, seed=runner.seed)),
        "real_qwen_medium_append": lambda p: runner.one(
            p, "real_qwen_medium_append", base_body(
                m, with_system(CLIENT_SYSTEM + "\n\n" + BRIEF_STYLE,
                               PROMPTS[p]["text"]),
                effort="medium", budget=2048, seed=runner.seed)),
        "real_bonsai_old_low": lambda p: runner.one(
            p, "real_bonsai_old_low", base_body(
                m, with_system(CLIENT_SYSTEM, PROMPTS[p]["text"]),
                effort="medium", budget=4096, seed=runner.seed)),
        "real_bonsai_low_prepend": lambda p: runner.one(
            p, "real_bonsai_low_prepend", base_body(
                m, with_system(FEWSHOT_STYLE + "\n\n" + CLIENT_SYSTEM,
                               PROMPTS[p]["text"]),
                effort="medium", budget=2560, seed=runner.seed)),
        "real_bonsai_low_append": lambda p: runner.one(
            p, "real_bonsai_low_append", base_body(
                m, with_system(CLIENT_SYSTEM + "\n\n" + FEWSHOT_STYLE,
                               PROMPTS[p]["text"]),
                effort="medium", budget=2560, seed=runner.seed)),
        "real_bonsai_medium_plain": lambda p: runner.one(
            p, "real_bonsai_medium_plain", base_body(
                m, with_system(CLIENT_SYSTEM, PROMPTS[p]["text"]),
                effort="medium", budget=2560, seed=runner.seed)),
        "real_bonsai_medium_append": lambda p: runner.one(
            p, "real_bonsai_medium_append", base_body(
                m, with_system(CLIENT_SYSTEM + "\n\n" + BRIEF_STYLE,
                               PROMPTS[p]["text"]),
                effort="medium", budget=2560, seed=runner.seed)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3.8-gguf-max")
    ap.add_argument("--suite", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--candidates", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--client-system-file", default=None,
                    help="real agent system prompt to put around the style")
    ap.add_argument("--out", default="/tmp/opencode/effort-ab.json")
    args = ap.parse_args()

    global CLIENT_SYSTEM
    if args.client_system_file:
        with open(args.client_system_file) as f:
            CLIENT_SYSTEM = f.read()
    runner = Runner(args.url, args.model, args.seed)
    cands = candidates(runner)
    selected = args.candidates or list(cands)
    missing = [c for c in selected if c not in cands]
    if missing:
        print(f"unknown candidates: {missing}", file=sys.stderr)
        return 2
    prompts = PILOT if args.suite == "pilot" else FULL
    rows = []
    for label in selected:
        for pid in prompts:
            r = cands[label](pid)
            rows.append(r)
            acc = "-" if r.get("accuracy") is None else ("OK" if r["accuracy"] else "FAIL")
            print(f"{label:16} {pid:15} {r.get('wall_s',0):6.1f}s "
                  f"reason={r.get('reasoning_tokens','-')!s:>5} "
                  f"ans={r.get('answer_tokens','-')!s:>5} "
                  f"plan={r.get('plan_tokens','-')!s:>4} "
                  f"{acc:4} {r.get('finish') or r.get('error','')}", flush=True)

    print("\n== summary ==")
    print(f"{'candidate':16} {'n':>3} {'med_reason':>10} {'med_total':>9} "
          f"{'med_wall':>8} {'accuracy':>8} {'cap-ish':>7}")
    for label in selected:
        rs = [r for r in rows if r["label"] == label and "error" not in r]
        if not rs:
            print(f"{label:16} errors")
            continue
        graded = [r for r in rs if r.get("accuracy") is not None]
        acc = sum(1 for r in graded if r["accuracy"]) / len(graded) if graded else 0
        med_reason = statistics.median(r.get("reasoning_tokens", 0) for r in rs)
        med_total = statistics.median(
            (r.get("reasoning_tokens", 0) + r.get("answer_tokens", 0) +
             r.get("plan_tokens", 0)) for r in rs)
        med_wall = statistics.median(r.get("wall_s", 0) for r in rs)
        cap = sum(1 for r in rs if r.get("finish") == "length")
        print(f"{label:16} {len(rs):3} {med_reason:10.0f} {med_total:9.0f} "
              f"{med_wall:8.1f} {acc:8.0%} {cap:7}")

    try:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "rows": rows}, f, indent=1)
        print(f"\nrows -> {args.out}")
    except OSError as e:
        print(f"could not write {args.out}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
