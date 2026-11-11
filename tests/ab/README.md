# A/B differential test — Python router vs .NET port

Compares the two implementations on the same request corpus. The default mode
is pure (no backend/docker/GPU): it runs the routing + effort pipeline of
`model-router.py` and of the .NET `src/` logic, then compares the result.

Compared per request:

- `profile` (routed tier, after speed-aware reuse)
- `note` (normalization log)
- `want_mmproj`
- `body` (the forwarded body, canonical JSON)
- `meta` (`effort`, `origin`, `template`, `budget`, `injected`, `toolLoop`, `style`)

## Run

```sh
python3 tests/ab/ab.py          # pure pipeline A/B
python3 tests/ab/ab.py --http   # also compare live /health and /v1/models
```

Exit code is 0 only when every case matches.

## Environment

- `MODEL_ROUTER_PY` — path to `model-router.py` (default: the sibling
  `../model-router/model-router.py`).
- `DOTNET` — dotnet executable (default: `dotnet`). On this machine the system
  `dotnet` may resolve to SDK 11 while `~/.dotnet` is read-only; use
  `/home/rafael.franca/.dotnet/dotnet` with a writable `DOTNET_CLI_HOME`.
- `AB_PY_PORT` / `AB_NET_PORT` — ports for `--http` (default 18191/18192).

## Corpus

Deterministic, fixed seed (`20261111`):

- every known model × effort (`none/minimal/low/medium/high/xhigh`) ×
  (`/v1/chat/completions`, `/v1/messages`);
- tools, media, `max_tokens`, `chat_template_kwargs`, explicit budget, system
  prompt;
- reuse across resident profiles (speed-aware tier reuse);
- explicit `final_profile` overrides;
- 80 randomized combinations.

The .NET side is `AbRunner`, a small console project that links the pure
`src/` files (ModelParser, Media, Profiles, Routing, Effort). It does not start
the relay or touch docker.

## `--http` mode

Starts both routers (`MODEL_ROUTER_PORT` set per process, no backend running)
and compares `/health` and `/v1/models`. This exercises the relay's static
surface end to end. The .NET side uses the published single-file binary when
present, otherwise `dotnet run`.
