#!/usr/bin/env bash
#
# scripts/setup.sh — Build the full model-serving stack from scratch.
#
# What it does:
#   1. Checks prerequisites (docker, nvidia-smi, python3, nvidia-docker runtime).
#   2. Ensures Docker images for every stack:
#        ninfer-4090:sm89                     (built from ~/Projetos/ThirdParty/ninfer-4090)
#        ghcr.io/ggml-org/llama.cpp:server-cuda  (pulled from ghcr.io)
#        bonsai-llama-4090:sm89               (built from bonsai-stack/llama.cpp submodule)
#   3. Checks model artifacts for all three stacks (see scripts/download-models.sh).
#   4. Installs systemd user unit: model-router.service
#   5. Starts/enables the router so model serving works on demand.
#   6. Verifies /health and runs each stack's check.sh.
#
# Safe to re-run: every step is idempotent.
#
# Manual steps NOT covered (must be done once per machine):
#   - Install Docker CE + nvidia-container-toolkit
#   - Add user to docker group: sudo usermod -aG docker $USER
#   - Download model artifacts (~48 GB total): scripts/download-models.sh all
set -euo pipefail
export LC_ALL=C LC_NUMERIC=C

# ---------------------------------------------------------------------------
# Paths (all relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd -- "${ROOT_DIR}/.." && pwd)"

# Stacks are git submodules (inside ROOT_DIR); fall back to sibling checkouts.
resolve_stack() {
  if [[ -d "${ROOT_DIR}/$1" ]]; then printf '%s' "${ROOT_DIR}/$1"
  else printf '%s' "${PARENT_DIR}/$1"; fi
}
NINFER_STACK="$(resolve_stack ninfer-stack)"
GGUF_STACK="$(resolve_stack gguf-stack)"
BONSAI_STACK="$(resolve_stack bonsai-stack)"
MODEL_ROUTER="${ROOT_DIR}"

NINFER_FORK="${NINFER_FORK:-${PARENT_DIR}/../ThirdParty/ninfer-4090}"
BONSAI_IMAGE="${NINFER_BONSAI_IMAGE:-bonsai-llama-4090:sm89}"

ROUTER_PORT="${MODEL_ROUTER_PORT:-18089}"
BACKEND_PORT="${NINFER_BACKEND_PORT:-8080}"
IDLE_SEC="${MODEL_ROUTER_IDLE_SEC:-900}"

UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_FILE="${UNIT_DIR}/model-router.service"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { printf '[setup] %s\n' "$*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'error: required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -e "$1" ]]; then
    printf 'error: required path missing: %s\n' "$1" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
log "Checking prerequisites..."
require_cmd docker
require_cmd nvidia-smi
require_cmd python3

if ! docker info >/dev/null 2>&1; then
  printf 'error: docker daemon is not reachable\n' >&2
  printf '       Start it: sudo systemctl start docker\n' >&2
  printf '       (If using Docker Desktop, open the app first)\n' >&2
  exit 1
fi

# Docker Desktop writes credsStore=desktop into ~/.docker/config.json. On a
# machine without that helper every pull/build fails while resolving image
# metadata ("error getting credentials"), so point this script at a clean
# config dir. Public images need no credentials at all.
docker_creds_broken() {
  local cfg="${HOME}/.docker/config.json" store
  [[ -f "$cfg" ]] || return 1
  store="$(sed -n 's/.*"credsStore"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$cfg" | head -1)"
  [[ -n "$store" ]] || return 1
  ! command -v "docker-credential-${store}" >/dev/null 2>&1
}
if docker_creds_broken; then
  export DOCKER_CONFIG="${TMPDIR:-/tmp}/model-router-docker"
  mkdir -p "$DOCKER_CONFIG"
  printf '{}\n' > "${DOCKER_CONFIG}/config.json"
  log "docker credential helper missing; using clean DOCKER_CONFIG=${DOCKER_CONFIG}"
fi

if ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  printf 'error: nvidia-smi cannot see a GPU\n' >&2
  exit 1
fi

# Verify GPU access from inside a container
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
  printf 'warning: docker --gpus all could not reach GPU\n' >&2
  printf '       Ensure nvidia-container-toolkit is installed:\n' >&2
  printf '       sudo dnf install nvidia-container-toolkit\n' >&2
  printf '       sudo nvidia-ctk runtime configure --runtime=docker\n' >&2
  printf '       sudo systemctl restart docker\n' >&2
fi
log "docker daemon ok; GPU visible"

# ---------------------------------------------------------------------------
# 2. Docker images
# ---------------------------------------------------------------------------
log "Ensuring docker images..."

LLAMA_IMAGE="${NINFER_LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b11854}"

if ! docker image inspect "${LLAMA_IMAGE}" >/dev/null 2>&1; then
  log "Pulling ${LLAMA_IMAGE} (first time, ~7GB)..."
  docker pull "${LLAMA_IMAGE}"
fi

if ! docker image inspect ninfer-4090:sm89 >/dev/null 2>&1; then
  if [[ ! -d "${NINFER_FORK}" ]]; then
    printf 'error: ninfer fork not found at %s\n' "${NINFER_FORK}" >&2
    printf '       Clone it: git clone https://github.com/UDPSendToFailed/ninfer-4090.git %s\n' "${NINFER_FORK}" >&2
    printf '       Or set NINFER_FORK=/path/to/ninfer-4090 and re-run.\n' >&2
    exit 1
  fi
  log "Building ninfer-4090:sm89 (this takes 15-30 min)..."
  ( cd "${NINFER_FORK}" && docker build -t ninfer-4090:sm89 . )
fi

# Bonsai: PrismML fork of llama.cpp, built from the submodule at the tagged
# gitlink. Stock llama.cpp cannot load the ternary packs and would emit
# garbage, so the label gate below is mandatory, not cosmetic.
if ! docker image inspect "${BONSAI_IMAGE}" >/dev/null 2>&1; then
  if [[ ! -f "${BONSAI_STACK}/llama.cpp/CMakeLists.txt" ]]; then
    printf 'error: bonsai-stack/llama.cpp submodule is not initialized\n' >&2
    printf '       Run: git -C %s submodule update --init --recursive\n' "${ROOT_DIR}" >&2
    exit 1
  fi
  bonsai_tag="$(git -C "${BONSAI_STACK}/llama.cpp" describe --tags --exact-match 2>/dev/null || printf 'unknown')"
  log "Building ${BONSAI_IMAGE} from bonsai-stack/llama.cpp (${bonsai_tag}, ~4 min)..."
  ( cd "${BONSAI_STACK}" && docker build -f Dockerfile \
      --build-arg PRISM_TAG="${bonsai_tag}" -t "${BONSAI_IMAGE}" . )
fi
if [[ -z "$(docker image inspect \
    -f '{{ index .Config.Labels "org.prismml.ternary" }}' \
    "${BONSAI_IMAGE}" 2>/dev/null)" ]]; then
  printf 'error: %s is not a PrismML ternary build\n' "${BONSAI_IMAGE}" >&2
  printf '       Stock llama.cpp mis-loads PTQ1_0/PQ2_0; rebuild it from bonsai-stack/.\n' >&2
  exit 1
fi

docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' \
  | grep -E 'ninfer-4090|llama.cpp|bonsai' || true

# ---------------------------------------------------------------------------
# 3. Model artifacts check
# ---------------------------------------------------------------------------
log "Checking model artifacts..."

NINFER_MODEL="${NINFER_STACK}/models/qwen3_8_27b.ninfer"
GGUF_BASE="${GGUF_STACK}/models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
GGUF_MTP="${GGUF_STACK}/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
GGUF_MM="${GGUF_STACK}/models/mmproj-Qwen3.8-27B-BF16.gguf"
BONSAI_BASE="${BONSAI_STACK}/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
BONSAI_MM="${BONSAI_STACK}/models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"

MISSING=0
missing() {
  printf 'warning: %s not found: %s\n' "$1" "$2" >&2
  MISSING=1
}
[[ -f "${NINFER_MODEL}" ]] || missing "ninfer model" "${NINFER_MODEL}"
[[ -f "${GGUF_BASE}" ]] || missing "GGUF base model" "${GGUF_BASE}"
[[ -f "${GGUF_MTP}" ]] || missing "GGUF MTP draft" "${GGUF_MTP}"
[[ -f "${GGUF_MM}" ]] || missing "GGUF vision projector" "${GGUF_MM}"
[[ -f "${BONSAI_BASE}" ]] || missing "Bonsai ternary weights" "${BONSAI_BASE}"
[[ -f "${BONSAI_MM}" ]] || missing "Bonsai vision projector" "${BONSAI_MM}"

if [[ "${MISSING}" -eq 1 ]]; then
  printf '\n' >&2
  printf 'Download the missing artifacts (~48 GB for all three stacks):\n' >&2
  printf '  %s/scripts/download-models.sh all\n' "${ROOT_DIR}" >&2
  printf '  # or one stack: ninfer | ninfer-vision | gguf | bonsai\n' >&2
fi

# ---------------------------------------------------------------------------
# 4. systemd user unit
# ---------------------------------------------------------------------------
log "Installing systemd user unit: model-router.service"
mkdir -p "${UNIT_DIR}"

cat > "${UNIT_FILE}" <<UNIT
[Unit]
Description=Model router (RTX 4090)

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=/usr/bin/python3 ${ROOT_DIR}/model-router.py
Environment=NINFER_SERVE_SH=${NINFER_STACK}/serve.sh
Environment=LLAMA_SERVE_SH=${GGUF_STACK}/serve-llama.sh
Environment=BONSAI_SERVE_SH=${BONSAI_STACK}/serve-bonsai.sh
Environment=NINFER_MODEL_DIR=${NINFER_STACK}/models
Environment=NINFER_GGUF_DIR=${GGUF_STACK}/models
Environment=NINFER_BONSAI_DIR=${BONSAI_STACK}/models
Environment=NINFER_BONSAI_IMAGE=${BONSAI_IMAGE}
Environment=MODEL_ROUTER_PORT=${ROUTER_PORT}
Environment=NINFER_BACKEND_PORT=${BACKEND_PORT}
Environment=MODEL_ROUTER_IDLE_SEC=${IDLE_SEC}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable model-router.service
# restart, not `enable --now`: on a re-run the unit file changed (new env) and
# an already-active service would keep the old environment.
systemctl --user restart model-router.service

# ---------------------------------------------------------------------------
# 5. Wait for router to be healthy
# ---------------------------------------------------------------------------
log "Waiting for router health check on :${ROUTER_PORT}..."
for i in $(seq 1 60); do
  if curl -s -m 5 "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1; then
    log "Router is healthy."
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    printf 'error: router did not become healthy in time\n' >&2
    journalctl --user -u model-router.service --no-pager -n 20 >&2 || true
    exit 1
  fi
  sleep 2
done

# ---------------------------------------------------------------------------
# 5b. Validate stacks
# ---------------------------------------------------------------------------
log "Running stack validation checks..."
bash "${ROOT_DIR}/check.sh" 2>&1 | tail -1
bash "${NINFER_STACK}/check.sh" 2>&1 | tail -1
bash "${GGUF_STACK}/check.sh" 2>&1 | tail -1
bash "${BONSAI_STACK}/check.sh" 2>&1 | tail -1

# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
log "Setup complete."
echo
echo "Router:   http://127.0.0.1:${ROUTER_PORT}/health"
echo "Backend:  http://127.0.0.1:${BACKEND_PORT}/health"
echo "Unit:     systemctl --user {start,stop,status} model-router.service"
echo "Logs:     journalctl --user -u model-router.service -f"
echo
echo "Stacks:"
echo "  ninfer-stack: ${NINFER_STACK}"
echo "  gguf-stack:   ${GGUF_STACK}"
echo "  bonsai-stack: ${BONSAI_STACK}"
echo "  model-router: ${ROOT_DIR}"
echo
echo "Pick a model in OpenCode:"
echo "  ninfer/qwen3.8-27b-64k    (64K ctx, int8)"
echo "  ninfer/qwen3.8-27b-128k   (128K ctx, rk8v4)"
echo "  ninfer/qwen3.8-27b-192k   (192K ctx, rk4v4-e8)"
echo "  ninfer/qwen3.8-27b-256k   (256K ctx, rk2v4-e8)"
echo "  llama.cpp/qwen3.8-gguf-speed   (64K, MTP, 92 tok/s)"
echo "  llama.cpp/qwen3.8-gguf-range   (96K ctx)"
echo "  llama.cpp/qwen3.8-gguf-max     (256K ctx)"
echo "  llama.cpp/qwen3.8-gguf-duo128  (2x128K parallel)"
echo "  llama.cpp/qwen3.8-gguf-quad    (4x64K parallel)"
echo "  llama.cpp/qwen3.8-gguf-single192  (192K ctx, MTP)"
echo "  prism/bonsai-2-27b-max      (ternary, 256K ctx)"
echo "  prism/bonsai-2-27b-duo256   (ternary, 2x256K parallel)"
echo "  prism/bonsai-2-27b-trio256  (ternary, 3x256K parallel, max)"
echo "  prism/bonsai-2-27b-duo128   (ternary, 2x128K)"
echo "  prism/bonsai-2-27b-quad128  (ternary, 4x128K)"
echo "  prism/bonsai-2-27b-octo64   (ternary, 8x64K)"
echo "  prism/bonsai-2-27b-vision   (ternary, 64K + mmproj)"
echo "  ninfer/qwen3.6-35b-vision     (35B vision, introspect images)"
echo
echo "Run in the foreground (debugging, no systemd):"
echo "  ${ROOT_DIR}/scripts/run-router.sh"
echo
echo "Tip: images in chat auto-trigger the vision backend; after IDLE_SEC"
echo "(${IDLE_SEC}s) of no requests the backend stops to free VRAM."
