# AGENT.md

Guidance for coding agents working in this repository.

## Release tags

- Tag system milestones with annotated semver tags: `vMAJOR.MINOR.PATCH`.
- `MINOR`: behavior or API change (routing, effort policy, backend
  contract). `PATCH`: backward-compatible fix.
- Tag `model-router` and `gguf-stack` together for the same milestone.
  Tag `bonsai-stack` too when its launcher or image pin moves.
- The `model-router` tag is the canonical system version: it pins the
  `gguf-stack` and `bonsai-stack` submodule commits.
- Never move or reuse a published tag. Push tags explicitly:
  `git push origin vX.Y.Z` and `git -C gguf-stack push origin vX.Y.Z`.
- Run `./check.sh` and a GGUF smoke test before tagging.

## Reasoning effort invariants

- Client tiers map to `EFFORT_POLICY` in `model-router.py`:
  `none`, `low=1536`, `medium=3072 (Qwen) / 6144 (Bonsai)`, `high=8192`,
  `xhigh=unbounded`.
- The budget is a safety net; the lever is the style in `EFFORT_STYLE`:
  Qwen `low` and `medium` get the same one-shot concise example (prepended);
  the budget does the tier split. Bonsai gets **no style**: its own real agent
  prompt is already concise, and with the example the open-ended task hit the
  cap (2,575 vs 1,885 without it). The template `low` sentence makes Bonsai
  loop; keep it on `medium`.
- Styles are prepended to the client's system prompt (or inserted as a system
  message), never replacing it; `inject_style` handles OpenAI string/list and
  Anthropic `system` string/list. The real-system A/B showed prepend 100% vs
  append 88% on Qwen (the needle); do not flip it.
- `low` caps at 1,536, below the measured realistic natural (Qwen 1,631,
  Bonsai 1,885), so it really is the lighter tier; the forced transition plus
  the answer reserve keeps the answer. `medium` keeps the larger budget.
- The effective budget is dynamic:
  `min(configured, max_tokens - ANSWER_RESERVE)` with reserve 1,536; below 256
  tokens of room it halves instead. `high`/`xhigh` are configured unbounded,
  so without the dynamic cap a 4k client request can spend all 4k thinking and
  return no answer.
- **The real agent system prompt matters**: with a 13 KB OpenCode prompt +
  `AGENT.md` + toolbelt the Qwen few-shot still cut the realistic from 1,890
  to 1,074 (plain final line, no labels; code tasks stay bare) and the median
  from 73 to 58, but the no-style Bonsai was better on the realistic (1,885 /
  22.2 s vs 2,575 / 29.0 s). Re-run
  `scripts/effort-probe.py --client-system-file <real-system>` when the client
  prompt changes.
- `low`/`medium` do **not** share a template any more on Qwen (`low` vs
  `medium`); on Bonsai both use `medium`. Changing effort mid-session changes
  the prompt prefix and re-prefills; the slot persistence only covers a
  backend restart.
- `high` maps to `xhigh` because the Qwen3.8 template rejects `high`;
  `minimal` aliases `low`.
- Budgets are llama.cpp-only (`reasoning_budget_tokens`); gate on `LLAMA_ALL`
  (GGUF_FILES | BONSAI_FILES). An unknown family falls back to the gguf table.
- `/v1/messages` uses `chat_template_kwargs` + `thinking.budget_tokens`;
  an explicit client `thinking` block wins, and an explicit client budget is
  never overwritten.
- The client catalog (`~/.config/opencode/opencode.jsonc`, v2 `variants`
  array) must list all six tiers per model. A tier missing there is
  unreachable from the UI even though the router supports it. Its default
  (no variant selected) is `low`.

## Backend notes

- Exactly one backend container may run: `backend_containers()` matches every
  stack image (the stock llama.cpp image and `bonsai-llama-4090`) and
  `stop_backend()` stops all of them before a new boot.
- The stock llama.cpp image is pinned in both `model-router.py` (`LLAMA_IMAGE`)
  and `gguf-stack/serve-llama.sh`. Bump both together; the pin is b12110
  (v0.9.0, 2026-11-04).
- The Bonsai image is built from the PrismML fork, which is the `llama.cpp/`
  submodule of `bonsai-stack`, checked out at the release tag
  `prism-b12128-d7b2896`. Its gitlink is the pin: never duplicate the commit
  in a file. Bump with `git -C llama.cpp checkout prism-bXXXXX-<sha>` and
  commit the new gitlink; pass the tag to the build only for provenance
  (`--build-arg PRISM_TAG="$(git -C llama.cpp describe --tags --exact-match)"`).
  The image carries the `org.prismml.ternary` label; `serve-bonsai.sh` refuses
  any image without it (stock llama.cpp mis-loads the packs and emits garbage).
- Slot persistence: the router saves idle slots (>= MODEL_ROUTER_SLOT_SAVE_MIN)
  before `stop_backend()` and restores them after a boot, keyed by a hash of
  the weights file. Hybrid models (both families) need the slot file's
  checkpoint appendix (upstream PR #26004): Bonsai requires prism-b12128+,
  Qwen b11854+. Without it the restore returns 200 and the next request still
  re-prefills everything ("forcing full prompt re-processing"). The `/slots`
  API is never proxied; it is a file-writing endpoint.
- `PROFILE_SLOTS` is load-bearing for concurrency: reuse requires the resident
  backend to have at least the requested slots, or requests that look
  parallel are queued inside llama-server.
- Reuse is family-scoped: `family()` maps each profile to `gguf` or
  `bonsai`, and `effective_profile()` never reuses a backend from another
  family. Same slot count is not permission to answer a Qwen request with
  Bonsai weights — that silently changes the model under the client.
- The router prefers the running backend over the requested tier to avoid
  a full re-prefill. A tier switch invalidates the prefix cache.
- The admission gate (`admit_prefill`) serializes arrivals behind an in-flight
  prefill so two prefills never interleave. It reads `/slots`, where
  `n_prompt_tokens` is **not** the prompt length: it is the KV cache for the
  task plus the batch queued for the next step, so it grows 2048, 4096, ...
  while a 100k-token prompt prefills. Prefill is `is_processing` with no token
  sampled yet and no tokens decoded; never infer it from a prompt-length
  subtraction. The gate is fail-open and must never be able to fail a request.
