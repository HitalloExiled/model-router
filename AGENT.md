# AGENT.md

Guidance for coding agents working in this repository.

## Release tags

- Tag system milestones with annotated semver tags: `vMAJOR.MINOR.PATCH`.
- `MINOR`: behavior or API change (routing, effort policy, backend
  contract). `PATCH`: backward-compatible fix.
- Tag `model-router` and `gguf-stack` together for the same milestone.
- The `model-router` tag is the canonical system version: it pins the
  `ninfer-stack` and `gguf-stack` submodule commits.
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
  to ninfer profiles; gate on `GGUF_FILES`.
- `/v1/messages` uses `chat_template_kwargs` + `thinking.budget_tokens`;
  an explicit client `thinking` block wins.
- The client catalog (`~/.config/opencode/opencode.jsonc`, v2 `variants`
  array) must list all six tiers per model. A tier missing there is
  unreachable from the UI even though the router supports it. Its default
  (no variant selected) is `low`.

## Backend notes

- The llama.cpp image is pinned in both `model-router.py` (`LLAMA_IMAGE`)
  and `gguf-stack/serve-llama.sh`. Bump both together.
- The router prefers the running backend over the requested tier to avoid
  a full re-prefill. A tier switch invalidates the prefix cache.
