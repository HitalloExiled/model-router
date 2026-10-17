#!/usr/bin/env bash
#
# scripts/setup.sh — Build the full NInfer model-serving stack from scratch.
#
# What it does:
#   1. Checks prerequisites (docker, nvidia-smi, nvidia-docker runtime).
#   2. Ensures Docker images:
#        ninfer-4090:sm89   (built from ~/Projetos/ThirdParty/ninfer-4090)
#        ghcr.io/ggml-org/llama.cpp:server-cuda (pulled from ghcr.io)
#   3. Installs systemd user unit: model-router.service
#   4. Starts/enables the router so model serving works on demand.
#   5. Verifies /health on the router port.
#
# Safe to re-run: every step is idempotent.
#
# Manual steps NOT covered (must be done once per machine):
#   - Install Docker CE + nvidia-container-toolkit
#   - Add user to docker group: sudo usermod -aG docker $USER
#   - Download model artifacts (17GB+ each):
#       scripts/download-qwen38.sh   (ninfer-stack)
#       curl -L ...qwen3_6_35b_a3b.ninfer   (ninfer-stack, optional vision)
#       curl -L ...qwen3_8_27B-GSQ-RCO-IQ3_S.gguf   (gguf-stack)
#       curl -L ...qwen3_8_27B-GSQ-RCO-IQ3_S-mtp.gguf   (gguf-stack)
#       curl -L ...mmproj-Qwen3.8-27B-BF16.gguf   (gguf-stack, vision)
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
MODEL_ROUTER="${ROOT_DIR}"

NINFER_FORK="${NINFER_FORK:-${PARENT_DIR}/../ThirdParty/ninfer-4090}"

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

if ! docker info >/dev/null 2>&1; then
  printf 'error: docker daemon is not reachable\n' >&2
  printf '       Start it: sudo systemctl start docker\n' >&2
  printf '       (If using Docker Desktop, open the app first)\n' >&2
  exit 1
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

if ! docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda >/dev/null 2>&1; then
  log "Pulling llama.cpp server image (first time, ~7GB)..."
  docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
fi

if ! docker image inspect ninfer-4090:sm89 >/dev/null 2>&1; then
  if [[ ! -d "${NINFER_FORK}" ]]; then
    printf 'error: ninfer fork not found at %s\n' "${NINFER_FORK}" >&2
    printf '       Clone it: git clone https://github.com/UDPSendToFailed/ninfer-4090.git %s\n' "${NINFER_FORK}" >&2
    exit 1
  fi
  log "Building ninfer-4090:sm89 (this takes 15-30 min)..."
  DOCKER_CONFIG="${DOCKER_CONFIG:-/tmp/docker-cfg}"
  mkdir -p "${DOCKER_CONFIG}"
  ( cd "${NINFER_FORK}" && DOCKER_CONFIG="${DOCKER_CONFIG}" docker build -t ninfer-4090:sm89 . )
fi

docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'ninfer-4090|llama.cpp'

# ---------------------------------------------------------------------------
# 3. Model artifacts check
# ---------------------------------------------------------------------------
log "Checking model artifacts..."

NINFER_MODEL="${NINFER_STACK}/models/qwen3_8_27b.ninfer"
GGUF_BASE="${GGUF_STACK}/models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
GGUF_MTP="${GGUF_STACK}/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
GGUF_MM="${GGUF_STACK}/models/mmproj-Qwen3.8-27B-BF16.gguf"

MISSING=0
if [[ ! -f "${NINFER_MODEL}" ]]; then
  printf 'warning: ninfer model not found: %s\n' "${NINFER_MODEL}" >&2
  printf '         Fetch with: ./scripts/download-qwen38.sh (in ninfer-stack)\n' >&2
  MISSING=1
fi
if [[ ! -f "${GGUF_BASE}" ]]; then
  printf 'warning: GGUF base model not found: %s\n' "${GGUF_BASE}" >&2
  MISSING=1
fi
if [[ ! -f "${GGUF_MTP}" ]]; then
  printf 'warning: GGUF MTP model not found: %s\n' "${GGUF_MTP}" >&2
  MISSING=1
fi
if [[ ! -f "${GGUF_MM}" ]]; then
  printf 'warning: GGUF vision projector not found: %s\n' "${GGUF_MM}" >&2
  MISSING=1
fi

if [[ "${MISSING}" -eq 1 ]]; then
  printf '\n' >&2
  printf 'Download models before using them:\n' >&2
  printf '  cd %s && ./scripts/download-qwen38.sh\n' "${NINFER_STACK}" >&2
  printf '  cd %s && ./scripts/download-gguf.sh\n' "${GGUF_STACK}" >&2
fi

# ---------------------------------------------------------------------------
# 3b. Generate download scripts if missing
# ---------------------------------------------------------------------------
if [[ ! -f "${GGUF_STACK}/scripts/download-gguf.sh" ]]; then
  log "Creating gguf-stack/download-gguf.sh stub..."
  mkdir -p "${GGUF_STACK}/scripts"
  cat > "${GGUF_STACK}/scripts/download-gguf.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
set -x
cd "$(dirname "$0")/.."
MODEL_DIR="models"
mkdir -p "${MODEL_DIR}"
echo "Downloading Qwen3.8-27B GSQ-RCO IQ3_S (base, 11.7 GB)..."
curl -L -C - --fail --output "${MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
  "https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
echo "Downloading Qwen3.8-27B GSQ-RCO IQ3_S-mtp (MTP draft, 12.1 GB)..."
curl -L -C - --fail --output "${MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
  "https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
echo "Downloading mmproj-Qwen3.8-27B-BF16.gguf (vision encoder, 889 MB)..."
curl -L -C - --fail --output "${MODEL_DIR}/mmproj-Qwen3.8-27B-BF16.gguf" \
  "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/mmproj-Qwen3.8-27B-BF16.gguf"
echo "All artifacts downloaded to ${MODEL_DIR}/"
EOF
  chmod +x "${GGUF_STACK}/scripts/download-gguf.sh"
fi

# ---------------------------------------------------------------------------
# 3c. Generate download-qwen38.sh if missing
# ---------------------------------------------------------------------------
if [[ ! -f "${NINFER_STACK}/scripts/download-qwen38.sh" ]]; then
  log "Creating ninfer-stack/scripts/download-qwen38.sh stub..."
  mkdir -p "${NINFER_STACK}/scripts"
  cat > "${NINFER_STACK}/scripts/download-qwen38.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR="models"
mkdir -p "${MODEL_DIR}"
echo "Downloading qwen3_8_27b.ninfer (17 GB)..."
curl -L -C - --fail --output "${MODEL_DIR}/qwen3_8_27b.ninfer" \
  "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer"
echo "Downloaded."
EOF
  chmod +x "${NINFER_STACK}/scripts/download-qwen38.sh"
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
Environment=NINFER_MODEL_DIR=${NINFER_STACK}/models
Environment=NINFER_GGUF_DIR=${GGUF_STACK}/models
Environment=MODEL_ROUTER_PORT=${ROUTER_PORT}
Environment=NINFER_BACKEND_PORT=${BACKEND_PORT}
Environment=MODEL_ROUTER_IDLE_SEC=${IDLE_SEC}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now model-router.service

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
bash "${NINFER_STACK}/check.sh" 2>&1 | tail -1
bash "${GGUF_STACK}/check.sh" 2>&1 | tail -1

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
echo "  ninfer/qwen3.6-35b-vision     (35B vision, introspect images)"
echo
echo "Tip: images in chat auto-trigger the vision backend; after IDLE_SEC"
echo "(${IDLE_SEC}s) of no requests the backend stops to free VRAM."
