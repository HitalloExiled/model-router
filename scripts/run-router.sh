#!/usr/bin/env bash
#
# scripts/run-router.sh — run the model router in the foreground.
#
# Same env the systemd unit uses, so behaviour is identical; handy for a
# terminal session, debugging, or a machine without systemd. Ctrl-C stops it
# (and the idle backends keep their normal 900 s unload).
#
# Usage:
#   scripts/run-router.sh [--port N] [--backend-port N] [--idle SEC]
#
# Prefer the unit for day-to-day use:
#   systemctl --user start model-router
#
# A second instance (--port) still shares backend :8080 and the docker
# containers, so use it for debugging, not side by side with the unit.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
parent="$(cd -- "$root/.." && pwd)"

stack() {
  if [[ -d "$root/$1" ]]; then printf '%s' "$root/$1"
  else printf '%s' "$parent/$1"; fi
}

port="${MODEL_ROUTER_PORT:-18089}"
backend_port="${NINFER_BACKEND_PORT:-8080}"
idle="${MODEL_ROUTER_IDLE_SEC:-900}"

while (( $# )); do
  case "$1" in
    --port)         port="${2:?--port needs a value}"; shift 2 ;;
    --port=*)       port="${1#--port=}"; shift ;;
    --backend-port) backend_port="${2:?--backend-port needs a value}"; shift 2 ;;
    --backend-port=*) backend_port="${1#--backend-port=}"; shift ;;
    --idle)         idle="${2:?--idle needs seconds}"; shift 2 ;;
    --idle=*)       idle="${1#--idle=}"; shift ;;
    -h|--help)      sed -n '2,13p' "$0"; exit 0 ;;
    *) printf 'Unknown arg: %s\n' "$1" >&2; exit 1 ;;
  esac
done

port_busy() {
  if command -v ss >/dev/null 2>&1; then
    [[ -n "$(ss -H -ltn "sport = :$1" 2>/dev/null)" ]]
  else
    ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
  fi
}

if port_busy "$port"; then
  printf 'error: port %s is already in use\n' "$port" >&2
  printf '       The systemd unit? systemctl --user status model-router\n' >&2
  printf '       Another shell? Try --port %s\n' "$((port + 1))" >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1 && ! docker info >/dev/null 2>&1; then
  printf 'warning: docker is not reachable; the router will start but cannot boot a backend\n' >&2
fi

export NINFER_SERVE_SH="${NINFER_SERVE_SH:-$(stack ninfer-stack)/serve.sh}"
export LLAMA_SERVE_SH="${LLAMA_SERVE_SH:-$(stack gguf-stack)/serve-llama.sh}"
export BONSAI_SERVE_SH="${BONSAI_SERVE_SH:-$(stack bonsai-stack)/serve-bonsai.sh}"
export NINFER_MODEL_DIR="${NINFER_MODEL_DIR:-$(stack ninfer-stack)/models}"
export NINFER_GGUF_DIR="${NINFER_GGUF_DIR:-$(stack gguf-stack)/models}"
export NINFER_BONSAI_DIR="${NINFER_BONSAI_DIR:-$(stack bonsai-stack)/models}"
export NINFER_BONSAI_IMAGE="${NINFER_BONSAI_IMAGE:-bonsai-llama-4090:sm89}"
export MODEL_ROUTER_PORT="$port"
export NINFER_BACKEND_PORT="$backend_port"
export MODEL_ROUTER_IDLE_SEC="$idle"

printf 'router:  http://127.0.0.1:%s (backend :%s, idle %ss)\n' \
  "$port" "$backend_port" "$idle"
printf 'logs:    this terminal; backend boot log: ~/.local/share/ninfer/ninfer-backend-boot.log\n'
printf 'stop:    Ctrl-C\n\n'

exec python3 "$root/model-router.py"
