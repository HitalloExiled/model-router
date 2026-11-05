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

Client tier -> template, style and thinking budget. The style is what makes
the model reason less; the budget is only the safety net:

| tier | Qwen template | Bonsai template | style | configured budget |
|---|---|---|---|---|
| `none` | thinking off | thinking off | — | — |
| `low` | `low` | `medium` | one-shot concise example | 3072 |
| `medium` | `medium` | `medium` | “think briefly” (Qwen) | 2048 / 6144 |
| `high` | `xhigh` | `xhigh` | — | 8192 |
| `xhigh` | `xhigh` | `xhigh` | — | unlimited |

`high` maps to `xhigh` because the template rejects `high`; `minimal` is an
alias for `low`. Budgets use llama.cpp's `reasoning_budget_tokens`, and the
style is prepended to the client's system prompt, never replacing it.

Why the styles are family-specific (measured 2026-11-05 on a 9-prompt battery,
Qwen b12110 / Bonsai prism-b12128):

- Qwen `low` with the template's low sentence + a one-example few-shot: the
  realistic task went 1,360 -> 991 reasoning tokens (42.6 s -> 30.5 s) at
  100% accuracy; median 134 -> 83.
- Qwen `medium` with a “think briefly” system prompt: 2,243 -> 1,225 tokens at
  100%.
- Bonsai `low` with the template's low sentence **looped** on the open-ended
  task (4,096 + no answer). The medium template + the same few-shot answers in
  2,052 tokens (24.1 s); the no-style medium answers in 1,685.
- `high`/`xhigh` used to run past the client's `max_tokens` and return no
  answer. The budget is now dynamic: `min(configured, max_tokens - 1536)`, so
  a 4k request thinks for 2,560 and keeps 1,536 for the answer (Qwen xhigh:
  4,096 + 0 -> 2,574 + 378). A tiny `max_tokens` halves instead of starving
  the answer.

The style/budget changes the prompt prefix, so changing effort mid-session
re-prefills the context; the slot persistence covers backend restarts, not a
template switch.

The client catalog (`~/.config/opencode/opencode.jsonc`) must list all six
tiers per model (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`); a tier
missing there is unreachable from the UI even though the router supports it.
The catalog default (no variant selected) is `low`.

The llama.cpp image is pinned to `server-cuda-b12110`; override with
`NINFER_LLAMA_IMAGE`.

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

## Slot persistence

Idle unload is 6 h by default (`MODEL_ROUTER_IDLE_SEC=21600`): the GPU has no
other tenant, and the old 15-minute timeout threw away a warm 100K-token
context on every pause, paying the full re-prefill on the next turn. A tier
switch still restarts the process; slot persistence makes that restart cheap.

Both launchers pass `--slot-save-path /slot-cache` and bind-mount
`slot-cache/`. Before `stop_backend()` the router saves every idle slot with
at least `MODEL_ROUTER_SLOT_SAVE_MIN` (default 2048) tokens; after the next
boot it restores the dump that belongs to the same weights file. Measured
cost of the thing it removes: 63.2 s for 115,491 tokens on `bonsai-trio200`
(~1,830 tok/s).

- The file name is `<family>-<sha1(weights-basename)[:10]>-slot<id>.bin`, so a
  Qwen dump is never handed to Bonsai, while tiers that share a weights file
  (`gguf-range/max/duo/quad`, every Bonsai tier) can hand a warm slot to each
  other.
- Hybrid models need the checkpoint appendix (upstream PR #26004). Bonsai
  requires a `prism-b12128+` image; Qwen a `b11854+` one. Without it restore
  answers 200 and the next request still re-prefills (the server logs
  "forcing full prompt re-processing due to lack of cache data").
- Budget: a 100K Bonsai slot is ~2.5 GiB of KV plus checkpoints, a 96K Qwen
  q8_0 slot ~14 GiB. `MODEL_ROUTER_SLOT_DISK_MIB` (default 32768) caps the
  directory, oldest first; a save is skipped when less than 8 GiB is free.
- The slot API is never proxied (403) and the backend is published on
  loopback only: it writes files.
- `MODEL_ROUTER_SLOT_PERSIST=0` disables the whole mechanism. Validate with
  `scripts/slot-persist-probe.py` (prime/save -> restart -> restore ->
  measure): a working restore leaves `cache_n` near the prompt length, not 0.

