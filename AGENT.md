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
  `none`, `low=512`, `medium=2048`, `high=8192`, `xhigh=unbounded`.
- `low`/`medium` share template `medium`; `high`/`xhigh` share template
  `xhigh`. This is deliberate: changing only the budget keeps the prompt
  prefix identical and preserves llama.cpp's prefix cache.
- `high` maps to `xhigh` because the Qwen3.8 template rejects `high`;
  `minimal` aliases `low`.
- Budgets are llama.cpp-only (`reasoning_budget_tokens`); gate on `LLAMA_ALL`
  (GGUF_FILES | BONSAI_FILES). An unknown family falls back to the gguf table.
- `/v1/messages` uses `chat_template_kwargs` + `thinking.budget_tokens`;
  an explicit client `thinking` block wins.
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
