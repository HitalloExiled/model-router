# model-router

OpenAI-compatible router (`:18089` -> backend `:8080`) with Ollama-style
load/unload: model name selects the backend profile, idle backends stop
themselves to free VRAM.

- `model-router.py` — routing, effort policy, SSE relay, retry, backend reuse
- `tests/test_router.py` — offline unit tests (no GPU/docker needed)

## Reasoning effort

Client tier -> Qwen3.8 template hint + hard thinking budget:

| tier | template | budget |
|---|---|---|
| `none` | thinking off | — |
| `low` | `medium` | 512 |
| `medium` | `medium` | 2048 |
| `high` | `xhigh` | 8192 |
| `xhigh` | `xhigh` | unlimited |

`high` maps to `xhigh` because the template rejects `high`; `minimal` is an
alias for `low`. Budgets use llama.cpp's `reasoning_budget_tokens` and are
injected only on GGUF profiles — `ninfer-serve` has no budget field, so its
tiers stay prompt-only. The llama.cpp image is pinned to
`server-cuda-b11854`; override with `NINFER_LLAMA_IMAGE`.

The backend stacks are git submodules of this repo:

- `ninfer-stack/` — NInfer `.ninfer` serving (`serve.sh`, `switch.sh`)
- `gguf-stack/`  — llama.cpp GGUF serving (`serve-llama.sh`)

Clone with submodules: `git clone --recurse-submodules <url>` (or
`git submodule update --init --recursive`).

Launchers are located via env (`NINFER_SERVE_SH`, `LLAMA_SERVE_SH`,
`NINFER_MODEL_DIR`, `NINFER_GGUF_DIR`); with no env set, each resolves inside
this repo first, then as a sibling checkout.
Run `./check.sh` to validate.
