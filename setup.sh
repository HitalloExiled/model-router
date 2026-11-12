#!/usr/bin/env bash
#
# setup.sh — install/refresh the model-router systemd user service (Python).
#
# Usage:
#   ./setup.sh [--print] [--no-start] [--port N] [--backend-port N] [--idle SEC]
#
# --print        print the unit and exit (do not touch systemd)
# --no-start     write the unit and enable it, but do not start/restart now
#
# Env overrides: MODEL_ROUTER_PORT, MODEL_ROUTER_BACKEND_PORT,
# MODEL_ROUTER_IDLE_SEC, MODEL_ROUTER_BONSAI_IMAGE.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_NAME="model-router.service"
UNIT_FILE="$UNIT_DIR/$UNIT_NAME"

PORT="${MODEL_ROUTER_PORT:-18089}"
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
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) printf 'Unknown arg: %s\n' "$1" >&2; exit 1 ;;
  esac
done

unit() {
  cat <<UNIT
[Unit]
Description=Model router (RTX 4090)

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=/usr/bin/python3 ${ROOT_DIR}/model-router.py
Environment=MODEL_ROUTER_LLAMA_SERVE_SH=${ROOT_DIR}/gguf-stack/serve-llama.sh
Environment=MODEL_ROUTER_BONSAI_SERVE_SH=${ROOT_DIR}/bonsai-stack/serve-bonsai.sh
Environment=MODEL_ROUTER_GGUF_DIR=${ROOT_DIR}/gguf-stack/models
Environment=MODEL_ROUTER_BONSAI_DIR=${ROOT_DIR}/bonsai-stack/models
Environment=MODEL_ROUTER_BONSAI_IMAGE=${BONSAI_IMAGE}
Environment=MODEL_ROUTER_PORT=${PORT}
Environment=MODEL_ROUTER_BACKEND_PORT=${BACKEND_PORT}
Environment=MODEL_ROUTER_IDLE_SEC=${IDLE_SEC}
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

unit > "$UNIT_FILE"
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"

if (( ! NO_START )); then
  systemctl --user restart "$UNIT_NAME"
  for _ in $(seq 1 30); do
    if curl -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      printf 'model-router healthy on :%s\n' "$PORT"
      exit 0
    fi
    sleep 1
  done
  printf 'error: model-router did not become healthy on :%s\n' "$PORT" >&2
  systemctl --user status "$UNIT_NAME" --no-pager >&2 || true
  exit 1
fi

printf 'model-router installed (start skipped)\n'
