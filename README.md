# model-router

OpenAI-compatible router (`:18089` -> backend `:8080`) with Ollama-style
load/unload: model name selects the backend profile, idle backends stop
themselves to free VRAM.

- `ninfer-router.py` — routing, effort injection, SSE relay, retry, backend reuse
- `tests/test_router.py` — offline unit tests (no GPU/docker needed)

The backend stacks are git submodules of this repo:

- `ninfer-stack/` — NInfer `.ninfer` serving (`serve.sh`, `switch.sh`)
- `gguf-stack/`  — llama.cpp GGUF serving (`serve-llama.sh`)

Clone with submodules: `git clone --recurse-submodules <url>` (or
`git submodule update --init --recursive`).

Launchers are located via env (`NINFER_SERVE_SH`, `LLAMA_SERVE_SH`,
`NINFER_MODEL_DIR`, `NINFER_GGUF_DIR`); with no env set, each resolves inside
this repo first, then as a sibling checkout.
Run `./check.sh` to validate.
