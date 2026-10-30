# AGENT.md

Guidance for coding agents working in this repository.

## Release tags

- Tag system milestones with annotated semver tags: `vMAJOR.MINOR.PATCH`.
- `MINOR`: behavior or API change (routing, effort policy, backend
  contract). `PATCH`: backward-compatible fix.
- Tag `model-router` and `gguf-stack` together for the same milestone.
  Tag `bonsai-stack` too when its launcher or image pin moves.
- The `model-router` tag is the canonical system version: it pins the
  `ninfer-stack`, `gguf-stack` and `bonsai-stack` submodule commits.
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
- Budgets are llama.cpp-only (`reasoning_budget_tokens`). Never send them
  to ninfer profiles; gate on `LLAMA_ALL` (GGUF_FILES | BONSAI_FILES).
- `/v1/messages` uses `chat_template_kwargs` + `thinking.budget_tokens`;
  an explicit client `thinking` block wins.
- The client catalog (`~/.config/opencode/opencode.jsonc`, v2 `variants`
  array) must list all six tiers per model. A tier missing there is
  unreachable from the UI even though the router supports it. Its default
  (no variant selected) is `low`.

## Backend notes

- Exactly one backend container may run: `backend_containers()` matches every
  stack image (`ninfer-4090`, the stock llama.cpp image, `bonsai-llama-4090`)
  and `stop_backend()` stops all of them before a new boot.
- The stock llama.cpp image is pinned in both `model-router.py` (`LLAMA_IMAGE`)
  and `gguf-stack/serve-llama.sh`. Bump both together.
- The Bonsai image is built from the PrismML fork, which is the `llama.cpp/`
  submodule of `bonsai-stack`, checked out at the release tag
  `prism-b10831-9d9efea`. Its gitlink is the pin: never duplicate the commit
  in a file. Bump with `git -C llama.cpp checkout prism-bXXXXX-<sha>` and
  commit the new gitlink; pass the tag to the build only for provenance
  (`--build-arg PRISM_TAG="$(git -C llama.cpp describe --tags --exact-match)"`).
  The image carries the `org.prismml.ternary` label; `serve-bonsai.sh` refuses
  any image without it (stock llama.cpp mis-loads the packs and emits garbage).
- `PROFILE_SLOTS` is load-bearing for concurrency: reuse requires the resident
  backend to have at least the requested slots, or requests that look
  parallel are queued inside llama-server.
- Reuse is family-scoped: `family()` maps each profile to `ninfer`, `gguf` or
  `bonsai`, and `effective_profile()` never reuses a backend from another
  family. Same slot count is not permission to answer a Qwen request with
  Bonsai weights — that silently changes the model under the client.
- The router prefers the running backend over the requested tier to avoid
  a full re-prefill. A tier switch invalidates the prefix cache.
