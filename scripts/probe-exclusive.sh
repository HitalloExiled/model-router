#!/usr/bin/env bash
# probe-exclusive.sh: "one model up, the other goes down".
#
# Requests a GGUF model and then a Bonsai model through the router and asserts
# that exactly ONE backend container is alive after each switch. A stale
# resident backend is what makes the next boot fail on VRAM.
#
# Usage: probe-exclusive.sh [--router URL] [--first MODEL] [--second MODEL]
set -euo pipefail

router="${MODEL_ROUTER_URL:-http://127.0.0.1:18089/v1}"
first="qwen3.8-gguf-range"
second="bonsai-2-27b-quad128"

while (( $# )); do
  case "$1" in
    --router) router="${2:?}"; shift 2 ;;
    --first) first="${2:?}"; shift 2 ;;
    --second) second="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) printf 'Unknown arg: %s\n' "$1" >&2; exit 1 ;;
  esac
done

ask() {
  local model="$1"
  curl -sS -m 900 -o /dev/null -w '%{http_code}\n' \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":4,\"stream\":false}" \
    "$router/chat/completions"
}

count() {
  docker ps --format '{{.Image}}' | grep -Ec 'llama\.cpp:server-cuda|bonsai-llama-4090|ninfer-4090' || true
}

rc=0
printf 'warming %s ...\n' "$first"
ask "$first" >/dev/null
n="$(count)"
printf 'containers after %s: %s\n' "$first" "$n"
[[ "$n" == "1" ]] || { printf 'FAIL: expected 1 backend container\n'; rc=1; }

printf 'switching to %s ...\n' "$second"
ask "$second" >/dev/null
n="$(count)"
printf 'containers after %s: %s\n' "$second" "$n"
[[ "$n" == "1" ]] || { printf 'FAIL: stale backend still resident\n'; rc=1; }

if [[ "$rc" == "0" ]]; then
  printf 'PASS: exclusive backend switching\n'
fi
exit "$rc"
