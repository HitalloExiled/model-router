# model-router

OpenAI-compatible router (`:18089` -> backend `:8080`) with Ollama-style
load/unload: model name selects the backend profile, idle backends stop
themselves to free VRAM.

- `ninfer-router.py` — routing, effort injection, SSE relay, retry, backend reuse
- `tests/test_router.py` — offline unit tests (no GPU/docker needed)

Backend launchers are located via env (`NINFER_SERVE_SH`, `LLAMA_SERVE_SH`,
`NINFER_MODEL_DIR`, `NINFER_GGUF_DIR`) with sibling-checkout defaults.
Run `./check.sh` to validate.
