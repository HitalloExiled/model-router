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
make publish    # AOT single-file ELF (src/ModelRouter/bin/Release/net11.0/.../publish/model-router)
make setup      # install/refresh model-router.net.service (:18090)
make run        # foreground
make e2e        # streaming chat against :18090 (needs backend/GPU)
make ab         # Python differential test (needs ../model-router)
```

`DOTNET ?= /usr/lib64/dotnet/dotnet` (the .NET 11 SDK). Override with
`make DOTNET=...`.

## Environment

The service is configured by `scripts/setup-service.sh`; it points
`MODEL_ROUTER_DIR` at the main checkout (the stacks are SSH-pinned submodules)
and runs the AOT binary from `src/ModelRouter/bin/Release/net11.0/linux-x64/publish/`.
