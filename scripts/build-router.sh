#!/usr/bin/env bash
#
# scripts/build-router.sh — publish the .NET 11 AOT router (single-file ELF).
#
# The output is what scripts/setup.sh installs in model-router.net.service and
# what scripts/run-router.sh execs:
#   bin/Release/net11.0/linux-x64/publish/model-router
#
# Env:
#   DOTNET   dotnet executable (default: dotnet)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

dotnet="${DOTNET:-dotnet}"
"$dotnet" publish src/ModelRouter/ModelRouter.csproj -c Release -r linux-x64

printf 'built: %s\n' "$root/src/ModelRouter/bin/Release/net11.0/linux-x64/publish/model-router"
