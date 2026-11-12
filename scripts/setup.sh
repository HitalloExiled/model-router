#!/usr/bin/env bash
#
# scripts/setup.sh — Build the full model-serving stack from scratch.
#
# What it does:
#   1. Checks prerequisites (docker, nvidia-smi, nvidia-docker runtime).
#   2. Ensures Docker images for both stacks:
#        ghcr.io/ggml-org/llama.cpp:server-cuda  (pulled from ghcr.io)
#        bonsai-llama-4090:sm89               (built from bonsai-stack/llama.cpp submodule)
#   3. Checks model artifacts for both stacks (see scripts/download-models.sh).
#   4. Installs systemd user unit: model-router.net.service
#   5. Starts/enables the router so model serving works on demand.
#   6. Verifies /health and runs each stack's check.sh.
#
# Safe to re-run: every step is idempotent.
#
# Manual steps NOT covered (must be done once per machine):
#   - Install Docker CE + nvidia-container-toolkit
#   - Add user to docker group: sudo usermod -aG docker $USER
#   - Download model artifacts (~40 GB total): scripts/download-models.sh all
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
  local d
  for d in "${ROOT_DIR}/stacks/$1" "${ROOT_DIR}/$1" "${PARENT_DIR}/$1" "${PARENT_DIR}/model-router/$1"; do
    if [[ -d "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then printf '%s' "$d"; return; fi
  done
  printf '%s' "${PARENT_DIR}/$1"
}
GGUF_STACK="$(resolve_stack gguf-stack)"
BONSAI_STACK="$(resolve_stack bonsai-stack)"
MODELS_DIR="${MODEL_ROUTER_MODELS_DIR:-$HOME/Drives/Files/AI/Models}"
GGUF_MODELS="${MODEL_ROUTER_GGUF_DIR:-$MODELS_DIR/gguf}"
BONSAI_MODELS="${MODEL_ROUTER_BONSAI_DIR:-$MODELS_DIR/bonsai}"
SLOT_DIR="${MODEL_ROUTER_SLOT_DIR:-$HOME/Drives/Files/.tmp/slot-cache}"
MODEL_ROUTER="${ROOT_DIR}"
# The router is the .NET 11 AOT port (single-file ELF); MODEL_ROUTER_BIN
# overrides the binary path.
ROUTER_BIN="${MODEL_ROUTER_BIN:-${ROOT_DIR}/src/ModelRouter/bin/Release/net11.0/linux-x64/publish/model-router}"

BONSAI_IMAGE="${MODEL_ROUTER_BONSAI_IMAGE:-bonsai-llama-4090:sm89}"

ROUTER_PORT="${MODEL_ROUTER_PORT:-18089}"
BACKEND_PORT="${MODEL_ROUTER_BACKEND_PORT:-8080}"
IDLE_SEC="${MODEL_ROUTER_IDLE_SEC:-21600}"

UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_FILE="${UNIT_DIR}/model-router.net.service"

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

LLAMA_IMAGE="${MODEL_ROUTER_LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b11854}"

if ! docker image inspect "${LLAMA_IMAGE}" >/dev/null 2>&1; then
  log "Pulling ${LLAMA_IMAGE} (first time, ~7GB)..."
  docker pull "${LLAMA_IMAGE}"
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
  | grep -E 'llama.cpp|bonsai' || true

# ---------------------------------------------------------------------------
# 3. Model artifacts check
# ---------------------------------------------------------------------------
log "Checking model artifacts..."

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
[[ -f "${GGUF_BASE}" ]] || missing "GGUF base model" "${GGUF_BASE}"
[[ -f "${GGUF_MTP}" ]] || missing "GGUF MTP draft" "${GGUF_MTP}"
[[ -f "${GGUF_MM}" ]] || missing "GGUF vision projector" "${GGUF_MM}"
[[ -f "${BONSAI_BASE}" ]] || missing "Bonsai ternary weights" "${BONSAI_BASE}"
[[ -f "${BONSAI_MM}" ]] || missing "Bonsai vision projector" "${BONSAI_MM}"

if [[ "${MISSING}" -eq 1 ]]; then
  printf '\n' >&2
  printf 'Download the missing artifacts (~40 GB for both stacks):\n' >&2
  printf '  %s/scripts/download-models.sh all\n' "${ROOT_DIR}" >&2
  printf '  # or one stack: gguf | bonsai\n' >&2
fi

# ---------------------------------------------------------------------------
# 4. systemd user unit
# ---------------------------------------------------------------------------
if [[ ! -x "${ROUTER_BIN}" ]]; then
  log "Building the .NET router (${ROUTER_BIN})..."
  bash "${SCRIPT_DIR}/build-router.sh"
fi

log "Installing systemd user unit: model-router.net.service"
mkdir -p "${UNIT_DIR}"

cat > "${UNIT_FILE}" <<UNIT
[Unit]
Description=Model router (.NET 11) — :18090

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${ROUTER_BIN}
Environment=MODEL_ROUTER_LLAMA_SERVE_SH=${GGUF_STACK}/serve-llama.sh
Environment=MODEL_ROUTER_BONSAI_SERVE_SH=${BONSAI_STACK}/serve-bonsai.sh
Environment=MODEL_ROUTER_GGUF_DIR=${GGUF_MODELS}
Environment=MODEL_ROUTER_BONSAI_DIR=${BONSAI_MODELS}
Environment=MODEL_ROUTER_SLOT_DIR=${SLOT_DIR}
Environment=MODEL_ROUTER_BONSAI_IMAGE=${BONSAI_IMAGE}
Environment=MODEL_ROUTER_PORT=${ROUTER_PORT}
Environment=MODEL_ROUTER_BACKEND_PORT=${BACKEND_PORT}
Environment=MODEL_ROUTER_IDLE_SEC=${IDLE_SEC}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable model-router.net.service
# restart, not `enable --now`: on a re-run the unit file changed (new env) and
# an already-active service would keep the old environment.
systemctl --user restart model-router.net.service

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
    journalctl --user -u model-router.net.service --no-pager -n 20 >&2 || true
    exit 1
  fi
  sleep 2
done

# ---------------------------------------------------------------------------
# 5b. Validate stacks
# ---------------------------------------------------------------------------
log "Running stack validation checks..."
bash "${ROOT_DIR}/check.sh" 2>&1 | tail -1
bash "${GGUF_STACK}/check.sh" 2>&1 | tail -1
bash "${BONSAI_STACK}/check.sh" 2>&1 | tail -1

# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
log "Setup complete."
echo
echo "Router:   http://127.0.0.1:${ROUTER_PORT}/health"
echo "Backend:  http://127.0.0.1:${BACKEND_PORT}/health"
echo "Unit:     systemctl --user {start,stop,status} model-router.net.service"
echo "Logs:     journalctl --user -u model-router.net.service -f"
echo
echo "Stacks:"
echo "  gguf-stack:   ${GGUF_STACK}"
echo "  bonsai-stack: ${BONSAI_STACK}"
echo "  model-router: ${ROOT_DIR}"
echo
echo "Pick a model in OpenCode:"
echo "  llama.cpp/qwen3.8-gguf-speed   (64K, MTP, 92 tok/s)"
echo "  llama.cpp/qwen3.8-gguf-range   (96K ctx)"
echo "  llama.cpp/qwen3.8-gguf-max     (256K ctx)"
echo "  llama.cpp/qwen3.8-gguf-duo128  (2x128K parallel)"
echo "  llama.cpp/qwen3.8-gguf-quad    (4x64K parallel)"
echo "  llama.cpp/qwen3.8-gguf-single192  (192K ctx, MTP)"
echo "  prism/bonsai-2-27b-max      (ternary, 256K ctx)"
echo "  prism/bonsai-2-27b-duo256   (ternary, 2x256K parallel)"
echo "  prism/bonsai-2-27b-trio256  (ternary, 3x256K parallel, max)"
echo "  prism/bonsai-2-27b-trio200  (ternary, 3x200K, ~4 GiB headroom)"
echo "  prism/bonsai-2-27b-duo128   (ternary, 2x128K)"
echo "  prism/bonsai-2-27b-quad128  (ternary, 4x128K)"
echo "  prism/bonsai-2-27b-penta128 (ternary, 5x128K)"
echo "  prism/bonsai-2-27b-hexa128  (ternary, 6x128K, max 128K)"
echo "  prism/bonsai-2-27b-octo64   (ternary, 8x64K)"
echo "  prism/bonsai-2-27b-vision   (ternary, 64K + mmproj)"
echo
echo "Run in the foreground (debugging, no systemd):"
echo "  ${ROOT_DIR}/scripts/run-router.sh"
echo
echo "Tip: images in chat load mmproj on the same tier; after IDLE_SEC"
echo "(${IDLE_SEC}s) of no requests the backend stops to free VRAM."
