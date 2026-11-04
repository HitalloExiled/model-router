# model-router

OpenAI-compatible router (`:18089` -> backend `:8080`) with Ollama-style
load/unload: model name selects the backend profile, idle backends stop
themselves to free VRAM.

- `model-router.py` — routing, effort policy, SSE relay, retry, backend reuse
- `tests/test_router.py` — offline unit tests (no GPU/docker needed)
- `scripts/setup.sh` — idempotent install of both backends + the unit
- `scripts/run-router.sh` — run the router in the foreground
- `scripts/download-models.sh` — fetch the weights per stack (resumable)

## Setup

```bash
git clone --recurse-submodules <url> model-router && cd model-router
./scripts/setup.sh          # images, model-artifact check, systemd unit
./scripts/download-models.sh all   # ~31 GB of weights (resumable)
```

`scripts/setup.sh` is idempotent: it pulls the pinned llama.cpp image, builds
`bonsai-llama-4090:sm89` when missing, installs and restarts
`model-router.service`, then runs every stack's `check.sh`.

To run the router in the foreground instead of the unit:

```bash
./scripts/run-router.sh                     # :18089, same env as the unit
./scripts/run-router.sh --port 18090 --idle 300
```

A second instance shares backend `:8080` and the docker containers, so it is a
debugging aid, not a replacement for the unit.

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
injected on every backend here (GGUF + Bonsai). The llama.cpp image is pinned
to `server-cuda-b11854`; override with `NINFER_LLAMA_IMAGE`.

The client catalog (`~/.config/opencode/opencode.jsonc`) must list all six
tiers per model (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`); a tier
missing there is unreachable from the UI even though the router supports it.
The catalog default (no variant selected) is `low`.

The backend stacks are git submodules of this repo:

- `gguf-stack/`  — llama.cpp GGUF serving (`serve-llama.sh`)
- `bonsai-stack/` — Bonsai 2 27B ternary GGUF (PrismML llama.cpp fork)

Exactly one backend is resident at a time: requesting any profile stops the
running container before booting the new one, whichever stack it belongs to.

Clone with submodules: `git clone --recurse-submodules <url>` (or
`git submodule update --init --recursive`).

Launchers are located via env (`MODEL_ROUTER_LLAMA_SERVE_SH`,
`MODEL_ROUTER_BONSAI_SERVE_SH`) and the artifact dirs the serve scripts read
(`NINFER_GGUF_DIR`, `NINFER_BONSAI_DIR`); with no env set, each resolves inside
this repo first, then as a sibling checkout. Run `./check.sh` to validate.

## Parallel slots

`PROFILE_SLOTS` records how many slots each tier boots. Reuse of the resident
backend requires it to have **at least** the requested slot count, otherwise a
1-slot tier would silently serialize requests aimed at a 4-slot one.
`MODEL_ROUTER_RESPECT_SLOTS=0` restores blind tier reuse. Measure with
`scripts/parallel-probe.py` (concurrency) and `scripts/probe-exclusive.sh`
(one model up, the other goes down).

Reuse is also restricted to the same model family (`gguf`, `bonsai`):
a resident Bonsai backend will never answer a `qwen3.8-gguf-*` request just
because the prompt fits and the slot count is sufficient.

The mmproj encoder is sticky: once a backend has it resident it also serves
text requests, so a mixed text/image session does not reboot a 20+ GiB
multi-slot backend (and pay a full re-prefill) on every turn. Only a media
request against a text-only backend triggers a reboot, to add the encoder.

## Admission gate (prefill vs decode)

A single large prefill blocks decode on every other slot for its whole
duration (`ceil(prompt / n_batch)` stalls), and two prefills running at the
same time pay a large throughput penalty on mixed-sequence attention. The
router decides *when* to hand a request over. The vendored llama.cpp is not
patched; this is router side only and fail-open.

Two rules, both measured from `/slots`:

- **phase A** (enforced) — an arrival waits while another slot is still
  prefilling. A slot counts as prefilling when it is processing a task of its
  own with at least `MODEL_ROUTER_PREFILL_GATE_MIN` (default 1024) prompt
  tokens in flight and no token sampled yet. `n_prompt_tokens` on its own is
  not enough: llama.cpp reports only the KV cache plus the batch queued for
  the next step, so during a prefill it grows 2048, 4096, ... instead of
  showing the whole prompt;
- **phase B** (opt-in, `MODEL_ROUTER_PREFILL_GATE_COLD=1`) — a *cold* large
  prompt (not a continuation of something just admitted,
  `>= MODEL_ROUTER_PREFILL_GATE_COLD_MIN`, default 8192 tokens) waits while
  any slot is decoding.

Phase A is on by default because it is a real win: with two concurrent
~10.7k-token cold prefills, the in-flight request finishes in 3.7 s instead of
7.6 s, and the queued one takes 8.5 s instead of 8.0 s. Phase B is off
because, with a long-lived decoder (the agent session itself), it waits out
the whole timeout and then admits the request into the same stall: no
throughput, only latency. `MODEL_ROUTER_PREFILL_GATE_ENFORCE=0` puts the gate
in shadow mode (log only).

Warm continuations are detected by hashing the request without its final
message: if that prefix was admitted recently, the turn is a cache hit and is
never held. Diagnostics are quiet by default (only holds are logged);
`MODEL_ROUTER_PREFILL_GATE_DIAG_VERBOSE=1` logs every admission, and
`MODEL_ROUTER_PREFILL_GATE_TIMEOUT` (default 10 s) bounds the wait so a stuck
prefill can never turn the router into a queue.

