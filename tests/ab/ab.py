#!/usr/bin/env python3
"""Differential A/B test: Python router vs the .NET port.

Feeds the same request corpus through both implementations' routing + effort
pipeline and compares (profile, note, want_mmproj, forwarded body, meta).
No backend/docker/GPU is needed for the default (pure) mode.

Usage:
    python3 tests/ab/ab.py            # pure pipeline A/B (default)
    python3 tests/ab/ab.py --http     # also compare live /health, /v1/models

Env:
    MODEL_ROUTER_PY   path to model-router.py (default: ../model-router)
    DOTNET            dotnet executable (default: dotnet)
    AB_PY_PORT / AB_NET_PORT   ports for --http (default: 18191/18192)
"""
import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY_DEFAULT = REPO.parent / "model-router" / "model-router.py"
WORK = HERE / ".work"


def load_py():
    path = Path(os.environ.get("MODEL_ROUTER_PY", str(PY_DEFAULT)))
    if not path.is_file():
        sys.exit(f"model-router.py not found: {path} (set MODEL_ROUTER_PY)")
    spec = importlib.util.spec_from_file_location("nr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_cases():
    cases = []
    P, M = "/v1/chat/completions", "/v1/messages"
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    tools = {"tools": [{"type": "function",
                        "function": {"name": "x",
                                     "parameters": {"type": "object",
                                                    "properties": {}}}}]}

    def add(label, model, path=P, effort=None, extra=None, current=None, final=None):
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
        if effort is not None:
            body["reasoning_effort"] = effort
        if extra:
            body.update(extra)
        cases.append({"label": label, "path": path, "body": json.dumps(body),
                      "currentProfile": current, "finalProfile": final})

    models = [
        "qwen3.8-gguf-max", "qwen3.8-gguf-range", "qwen3.8-gguf-speed",
        "qwen3.8-gguf-duo128", "qwen3.8-gguf-quad", "qwen3.8-gguf-single200",
        "qwen3.8-gguf-vision", "bonsai-2-27b-max", "bonsai-2-27b-duo256",
        "bonsai-2-27b-trio200", "bonsai-2-27b-octo64", "bonsai-2-27b-quad128",
        "bonsai-2-27b-vision", "mystery-model",
    ]
    efforts = [None, "none", "minimal", "low", "medium", "high", "xhigh"]
    for model in models:
        for effort in efforts:
            for path in (P, M):
                add(f"{model}|{path}|{effort}", model, path=path, effort=effort)

    for model in ("qwen3.8-gguf-max", "bonsai-2-27b-trio200", "qwen3.8-gguf-range"):
        add(f"{model}|tools", model, extra=tools)
        add(f"{model}|media", model,
            extra={"messages": [{"role": "user", "content": [img]}]})
        for mt in (512, 1024, 4096, 8192):
            add(f"{model}|max{mt}", model, extra={"max_tokens": mt})
        add(f"{model}|off", model,
            extra={"chat_template_kwargs": {"enable_thinking": False}})
        add(f"{model}|budget", model, extra={"reasoning_budget_tokens": 999})
        add(f"{model}|system", model,
            extra={"messages": [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "hi"}]})

    for model in ("qwen3.8-gguf-max", "bonsai-2-27b-quad128"):
        add(f"{model}|m-thinking", model, path=M,
            extra={"thinking": {"type": "disabled"}})
        add(f"{model}|m-budget", model, path=M,
            extra={"thinking": {"type": "enabled", "budget_tokens": 4096}})
        add(f"{model}|m-tools", model, path=M, extra=tools)

    # Reuse (resident backend profile): the pipeline may keep the current tier.
    for current in ("gguf-max", "gguf-range", "gguf-speed",
                    "bonsai-trio200", "bonsai-max", "gguf-quad"):
        for model in ("qwen3.8-gguf-max", "qwen3.8-gguf-range",
                      "qwen3.8-gguf-single200", "bonsai-2-27b-trio256",
                      "bonsai-2-27b-duo128", "bonsai-2-27b-octo64"):
            add(f"{model}|cur={current}", model, current=current)

    # Explicit final-profile override (the effort policy follows the resident).
    for fp in ("bonsai-max", "gguf-max", "bonsai-trio200"):
        for model in ("qwen3.8-gguf-max-low", "bonsai-2-27b-quad128-low"):
            add(f"{model}|final={fp}", model, final=fp)

    # Randomized corpus, fixed seed for reproducibility.
    rnd = random.Random(20261111)
    for i in range(80):
        model = rnd.choice(models)
        path = rnd.choice((P, M))
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
        if rnd.random() < 0.5:
            body["reasoning_effort"] = rnd.choice(
                ["none", "minimal", "low", "medium", "high", "xhigh"])
        if rnd.random() < 0.4:
            body["max_tokens"] = rnd.choice([512, 1024, 2048, 4096, 8192])
        if rnd.random() < 0.3:
            body.update(tools)
        if rnd.random() < 0.2:
            body["chat_template_kwargs"] = {
                "enable_thinking": rnd.choice([True, False])}
        current = rnd.choice([None, "gguf-max", "gguf-range",
                              "bonsai-trio200", "bonsai-octo64", "gguf-speed"])
        cases.append({"label": f"rand{i}", "path": path, "body": json.dumps(body),
                      "currentProfile": current, "finalProfile": None})
    return cases


def reference(nr, cases):
    refs = []
    for case in cases:
        decoded = json.loads(case["body"])
        route = nr.route_request(decoded)
        requested = route[0]
        need_vision = nr.has_media(decoded)
        est = len(case["body"].encode()) / 3.5
        reused = nr.effective_profile(requested, case["currentProfile"], est,
                                      need_vision)
        target = case["finalProfile"] if case["finalProfile"] is not None else reused
        prof, new_body, note, mm, meta = nr.normalize_chat_body(
            case["path"], case["body"].encode(), route=route,
            final_profile=target)
        refs.append({
            "label": case["label"],
            "profile": prof,
            "note": note,
            "wantMmproj": mm,
            "body": canon(json.loads(new_body)),
            "meta": {
                "effort": meta.get("effort"),
                "origin": meta.get("origin"),
                "template": meta.get("template"),
                "budget": meta.get("budget"),
                "injected": bool(meta.get("injected_budget")),
                "toolLoop": bool(meta.get("tool_loop")),
                "style": bool(meta.get("style")),
            },
        })
    return refs


def run_dotnet(cases):
    WORK.mkdir(parents=True, exist_ok=True)
    cases_path = WORK / "cases.json"
    out_path = WORK / "net.jsonl"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    dotnet = os.environ.get("DOTNET", "dotnet")
    cmd = [dotnet, "run", "--project", str(HERE / "AbRunner"), "-c", "Release",
           "--", str(cases_path), str(out_path)]
    subprocess.run(cmd, check=True, cwd=str(WORK))
    return [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]


def compare(refs, got):
    if len(refs) != len(got):
        print(f"COUNT MISMATCH: python={len(refs)} net={len(got)}")
        return False
    fails = 0
    for ref, net in zip(refs, got):
        if ref != net:
            fails += 1
            if fails <= 20:
                print(f"--- FAIL {ref['label']}")
                for key in sorted(set(ref) | set(net)):
                    if ref.get(key) != net.get(key):
                        print(f"    {key}: py={ref.get(key)!r} net={net.get(key)!r}")
    print(f"\nA/B pipeline: OK={len(refs) - fails} FAIL={fails} "
          f"(total {len(refs)})")
    return fails == 0


def http_ab(py_path):
    import socket
    py_port = int(os.environ.get("AB_PY_PORT", "18191"))
    net_port = int(os.environ.get("AB_NET_PORT", "18192"))

    def free(port):
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    if not free(py_port) or not free(net_port):
        print("HTTP A/B: ports busy, skipping")
        return True

    dotnet = os.environ.get("DOTNET", "dotnet")
    py = subprocess.Popen([sys.executable, str(py_path)],
                          env={**os.environ, "MODEL_ROUTER_PORT": str(py_port)},
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    binary = REPO / "bin/Release/net10.0/linux-x64/publish/model-router"
    if binary.exists():
        net_cmd = [str(binary)]
    else:
        net_cmd = [dotnet, "run", "--project", str(REPO), "-c", "Release",
                   "--no-launch-profile"]
    net = subprocess.Popen(net_cmd,
                           env={**os.environ, "MODEL_ROUTER_PORT": str(net_port)},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def get(port, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return json.loads(r.read())

    def wait(port):
        for _ in range(180):
            try:
                get(port, "/health")
                return True
            except Exception:
                time.sleep(0.5)
        return False

    try:
        ok = True
        if not wait(py_port):
            print("HTTP A/B: python router did not start")
            ok = False
        if not wait(net_port):
            print("HTTP A/B: .NET router did not start")
            ok = False
        if ok:
            for path in ("/health", "/v1/models"):
                a = get(py_port, path)
                b = get(net_port, path)
                same = a == b
                print(f"HTTP {path}: {'OK' if same else 'FAIL'}")
                if not same:
                    ok = False
                    print("  py :", json.dumps(a, sort_keys=True)[:400])
                    print("  net:", json.dumps(b, sort_keys=True)[:400])
        return ok
    finally:
        for proc in (py, net):
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true",
                        help="also compare live /health and /v1/models")
    args = parser.parse_args()

    nr, py_path = load_py()
    cases = build_cases()
    refs = reference(nr, cases)
    got = run_dotnet(cases)
    ok = compare(refs, got)
    if args.http:
        ok = http_ab(py_path) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
