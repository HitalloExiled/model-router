#!/usr/bin/env bash
#
# scripts/download-models.sh — fetch every model artifact the stacks serve.
#
# Artifacts land in each stack's models/ (the exact paths the launchers expect),
# so this is the single command to run after a fresh clone. Downloads resume
# with curl -C -, so re-running after an interruption is safe.
#
# Usage:
#   scripts/download-models.sh [ninfer|ninfer-vision|gguf|bonsai|all]
#
#   ninfer        qwen3_8_27b.ninfer            (~17 GB)
#   ninfer-vision qwen3_6_35b_a3b.ninfer        (~21 GB, optional 35B vision)
#   gguf          IQ3_S base + MTP + mmproj      (~24 GB)
#   bonsai        Bonsai 2 27B PTQ1_0 + mmproj   (~6.2 GB, ternary)
#   all           ninfer + gguf + bonsai (default)
#
# Disk needed for `all`: ~48 GB. Override the destination of a stack with
# NINFER_MODEL_DIR / NINFER_GGUF_DIR / NINFER_BONSAI_DIR.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
parent="$(cd -- "$root/.." && pwd)"

stack() {
  if [[ -d "$root/$1" ]]; then printf '%s' "$root/$1"
  else printf '%s' "$parent/$1"; fi
}

ninfer_dir="${NINFER_MODEL_DIR:-$(stack ninfer-stack)/models}"
gguf_dir="${NINFER_GGUF_DIR:-$(stack gguf-stack)/models}"
bonsai_dir="${NINFER_BONSAI_DIR:-$(stack bonsai-stack)/models}"

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

get_ninfer() {
  local script
  script="$(stack ninfer-stack)/download-qwen38.sh"
  if [[ -f "$script" ]]; then
    NINFER_MODEL_DIR="$ninfer_dir" bash "$script"
  else
    fetch "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer" \
      "$ninfer_dir/qwen3_8_27b.ninfer" "Qwen3.8-27B NInfer (~17 GB)"
  fi
}

get_ninfer_vision() {
  local script
  script="$(stack ninfer-stack)/download-qwen36-35b-vision.sh"
  if [[ -f "$script" ]]; then
    NINFER_MODEL_DIR="$ninfer_dir" bash "$script"
  else
    fetch "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c/qwen3_6_35b_a3b.ninfer" \
      "$ninfer_dir/qwen3_6_35b_a3b.ninfer" "Qwen3.6-35B-A3B NInfer vision (~21 GB)"
  fi
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
  ninfer)        get_ninfer ;;
  ninfer-vision) get_ninfer_vision ;;
  gguf)          get_gguf ;;
  bonsai)        get_bonsai ;;
  all)           get_ninfer; get_gguf; get_bonsai ;;
  -h|--help)     sed -n '2,19p' "$0"; exit 0 ;;
  *) printf 'Unknown target: %s (want ninfer|ninfer-vision|gguf|bonsai|all)\n' "$what" >&2
     exit 1 ;;
esac

printf '\nArtifacts:\n'
du -sh -- "$ninfer_dir" "$gguf_dir" "$bonsai_dir" 2>/dev/null || true
