#!/usr/bin/env bash
#
# scripts/setup-service.sh — install/refresh the .NET router systemd user service.
#
#   model-router.net.service  ->  .NET router on :18090 (AOT binary)
#
# Usage:
#   scripts/setup-service.sh [--print] [--no-start] [--port N] [--backend-port N] [--idle SEC]
#
# --print        print the unit and exit (do not touch systemd)
# --no-start     write the unit and enable it, but do not start/restart now
#
# The .NET router and the stacks live in the same checkout (RouterDir);
# MODEL_ROUTER_MAIN_DIR overrides it (e.g. a worktree with no submodules).
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_DIR="${MODEL_ROUTER_MAIN_DIR:-$ROOT_DIR}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_NAME="model-router.net.service"
UNIT_FILE="$UNIT_DIR/$UNIT_NAME"
DOTNET="${DOTNET:-/usr/lib64/dotnet/dotnet}"
AOT_BIN="${ROOT_DIR}/src/ModelRouter/bin/Release/net11.0/linux-x64/publish/model-router"
MANAGED_DLL="${ROOT_DIR}/src/ModelRouter/bin/Release/net11.0/model-router.dll"

PORT="${MODEL_ROUTER_PORT:-18090}"
BACKEND_PORT="${MODEL_ROUTER_BACKEND_PORT:-8080}"
IDLE_SEC="${MODEL_ROUTER_IDLE_SEC:-21600}"
BONSAI_IMAGE="${MODEL_ROUTER_BONSAI_IMAGE:-bonsai-llama-4090:sm89}"
PRINT=0
NO_START=0

while (( $# )); do
  case "$1" in
    --print) PRINT=1; shift ;;
    --no-start) NO_START=1; shift ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --port=*) PORT="${1#--port=}"; shift ;;
    --backend-port) BACKEND_PORT="${2:?--backend-port needs a value}"; shift 2 ;;
    --backend-port=*) BACKEND_PORT="${1#--backend-port=}"; shift ;;
    --idle) IDLE_SEC="${2:?--idle needs a value}"; shift 2 ;;
    --idle=*) IDLE_SEC="${1#--idle=}"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) printf 'Unknown arg: %s\n' "$1" >&2; exit 1 ;;
  esac
done

if [[ -x "$AOT_BIN" ]]; then
  EXEC_START="$AOT_BIN"
else
  EXEC_START="$DOTNET $MANAGED_DLL"
fi

unit() {
  cat <<UNIT
[Unit]
Description=Model router (.NET 11) — :${PORT}

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${EXEC_START}
Environment=MODEL_ROUTER_PORT=${PORT}
Environment=MODEL_ROUTER_BACKEND_PORT=${BACKEND_PORT}
Environment=MODEL_ROUTER_IDLE_SEC=${IDLE_SEC}
Environment=MODEL_ROUTER_DIR=${MAIN_DIR}
Environment=MODEL_ROUTER_LLAMA_SERVE_SH=${MAIN_DIR}/stacks/gguf-stack/serve-llama.sh
Environment=MODEL_ROUTER_BONSAI_SERVE_SH=${MAIN_DIR}/stacks/bonsai-stack/serve-bonsai.sh
Environment=MODEL_ROUTER_BONSAI_IMAGE=${BONSAI_IMAGE}
Environment=DOTNET_ROOT=/usr/lib64/dotnet
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
}

if (( PRINT )); then
  unit
  exit 0
fi

mkdir -p "$UNIT_DIR" 2>/dev/null || true
if [[ ! -w "$UNIT_DIR" ]]; then
  printf 'error: cannot write the user unit dir: %s\n' "$UNIT_DIR" >&2
  printf '       run this from a normal shell (the agent sandbox may be read-only)\n' >&2
  exit 1
fi

if [[ ! -x "$AOT_BIN" && ! -f "$MANAGED_DLL" ]]; then
  printf 'error: router not built (missing %s and %s)\n' "$AOT_BIN" "$MANAGED_DLL" >&2
  printf '       build it with: %s/scripts/build-router.sh\n' "$ROOT_DIR" >&2
  exit 1
fi

unit > "$UNIT_FILE"
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"

if (( ! NO_START )); then
  systemctl --user restart "$UNIT_NAME"
  for _ in $(seq 1 30); do
    if curl -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      printf 'model-router.net healthy on :%s\n' "$PORT"
      exit 0
    fi
    sleep 1
  done
  printf 'error: model-router.net did not become healthy on :%s\n' "$PORT" >&2
  systemctl --user status "$UNIT_NAME" --no-pager >&2 || true
  exit 1
fi

printf 'model-router.net installed (start skipped)\n'
