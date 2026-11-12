# model-router.net

.NET 11 AOT port of the model router. It listens on `:18090`
(`model-router.net.service`) and forwards to a single llama.cpp/Bonsai backend
on `:8080`, with model routing, the per-family effort policy, slot-cache
persistence and the prefill admission gate.

## Layout

```
Makefile
scripts/                 build/run/setup/download launchers
src/ModelRouter/         router (Program.cs + pure logic + BackendManager)
src/ModelRouter.Tests/   xUnit v3 unit tests
stacks/                  gguf-stack / bonsai-stack submodules
probes/                  .NET file-based apps (effort, cache-reuse, parallel,
                         prefill-bench, prefill-probe, slot-persist, e2e)
```

## Build / test / run

```
make build      # dotnet build (net11)
make test       # xUnit v3 unit tests
make analyze    # roslynator analyze on ModelRouter.slnx
make publish    # AOT single-file ELF (src/ModelRouter/bin/Release/net11.0/.../publish/model-router)
make setup      # install/refresh model-router.net.service (:18090)
make run        # foreground
make e2e        # streaming chat against :18090 (needs backend/GPU)
```

`DOTNET ?= /usr/lib64/dotnet/dotnet` (the .NET 11 SDK). Override with
`make DOTNET=...`. The Makefile exports `DOTNET_ROOT` derived from `DOTNET`:
`roslynator` discovers MSBuild through `DOTNET_ROOT`, and pointing it at an
older SDK (e.g. `~/.dotnet` with 10.0.203) makes every reference fail to
resolve, which surfaces as thousands of bogus CS0518/CS0103 diagnostics.
Use `make analyze`, or run
`DOTNET_ROOT=/usr/lib64/dotnet dotnet roslynator analyze ModelRouter.slnx`
directly.

## Environment

The service is configured by `scripts/setup-service.sh`; it points
`MODEL_ROUTER_DIR` at the repo root and runs the AOT binary from
`src/ModelRouter/bin/Release/net11.0/linux-x64/publish/`. The stacks under
`stacks/` are SSH-pinned submodules: `git submodule update --init` before the
first `make setup` on a fresh clone.

Model artifacts and the slot cache (warm KV state, several GB) default to the
repo: the stack launchers read `<repo>/stacks/<stack>/models` and the router
keeps slots in `<repo>/slot-cache` (mounted into the backend at `/slot-cache`).

A machine that stores them elsewhere (e.g. another drive) keeps the paths
outside the repo in `~/.config/model-router/env`:

```
MODEL_ROUTER_GGUF_DIR=/path/to/gguf
MODEL_ROUTER_BONSAI_DIR=/path/to/bonsai
MODEL_ROUTER_SLOT_DIR=/path/to/slot-cache
```

The scripts load it through `scripts/lib/env.sh` (`MODEL_ROUTER_ENV_FILE`
overrides the path) and the unit reads it with
`EnvironmentFile=-%h/.config/model-router/env`. Anything already exported wins
over the file. Keeping these paths out of the repo also means a submodule
`deinit`/clone cannot destroy them.

The Python implementation this port replaced is kept only in history
(`backup/main-python`).
