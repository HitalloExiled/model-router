#!/usr/bin/env bash
#
# scripts/download-models.sh — fetch every model artifact the stacks serve.
#
# Artifacts land in each stack's models/ (the exact paths the launchers expect),
# so this is the single command to run after a fresh clone. Downloads resume
# with curl -C -, so re-running after an interruption is safe.
#
# Usage:
#   scripts/download-models.sh [gguf|bonsai|all]
#
#   gguf          IQ3_S base + MTP + mmproj      (~24 GB)
#   bonsai        Bonsai 2 27B PTQ1_0 + mmproj   (~6.2 GB, ternary)
#   all           gguf + bonsai (default)
#
# Disk needed for `all`: ~31 GB. Override the destination of a stack with
# MODEL_ROUTER_GGUF_DIR / MODEL_ROUTER_BONSAI_DIR (the names the serve scripts read).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
parent="$(cd -- "$root/.." && pwd)"

stack() {
  if [[ -d "$root/$1" ]]; then printf '%s' "$root/$1"
  else printf '%s' "$parent/$1"; fi
}

gguf_dir="${MODEL_ROUTER_GGUF_DIR:-$(stack gguf-stack)/models}"
bonsai_dir="${MODEL_ROUTER_BONSAI_DIR:-$(stack bonsai-stack)/models}"

fetch() {  # fetch <url> <dest> <label>
  local url="$1" dest="$2" label="$3"
  mkdir -p -- "$(dirname -- "$dest")"
  printf '==> %s\n' "$label"
  if ! curl -L -C - --fail --output "$dest" "$url"; then
    printf 'Download failed: %s\nRe-run this script to resume.\n' "$dest" >&2
    exit 1
  fi
  printf '    ok: %s\n' "$dest"
}

get_gguf() {
  local repo="https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main"
  fetch "$repo/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
    "$gguf_dir/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
    "Qwen3.8-27B IQ3_S base (~11 GB)"
  fetch "$repo/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
    "$gguf_dir/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
    "Qwen3.8-27B IQ3_S MTP draft (~12 GB)"
  fetch "$repo/mmproj-Qwen3.8-27B-BF16.gguf" \
    "$gguf_dir/mmproj-Qwen3.8-27B-BF16.gguf" \
    "Qwen3.8-27B mmproj BF16 (~0.9 GB)"
}

get_bonsai() {
  local repo="https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main"
  fetch "$repo/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
    "$bonsai_dir/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
    "Bonsai 2 27B ternary PTQ1_0 (~5.95 GB)"
  fetch "$repo/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf" \
    "$bonsai_dir/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf" \
    "Bonsai 2 27B mmproj Q8_0 (~0.63 GB)"
}

what="${1:-all}"
case "$what" in
  gguf)          get_gguf ;;
  bonsai)        get_bonsai ;;
  all)           get_gguf; get_bonsai ;;
  -h|--help)     sed -n '2,17p' "$0"; exit 0 ;;
  *) printf 'Unknown target: %s (want gguf|bonsai|all)\n' "$what" >&2
     exit 1 ;;
esac

printf '\nArtifacts:\n'
du -sh -- "$gguf_dir" "$bonsai_dir" 2>/dev/null || true
