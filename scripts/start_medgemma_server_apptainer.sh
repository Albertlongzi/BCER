#!/bin/bash
set -euo pipefail

# Start an OpenAI-compatible MedGemma server inside apptainer (Transformers backend).
#
# Usage example:
#   module load apptainer/1.0.1
#   export HOST_REPO=/path/to/MRI_Agent
#   export HOST_COMMON_DATA=/path/to/shared_cache
#   export MODEL_ID=google/medgemma-1.5-4b-it
#   export PORT=8000
#   bash MRI_Agent/scripts/start_medgemma_server_apptainer.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST_REPO="${HOST_REPO:-${DEFAULT_REPO}}"
HOST_COMMON_DATA="${HOST_COMMON_DATA:-${HOST_REPO}}"
IMG="${IMG:-${HOST_COMMON_DATA}/vllm_latest.sif}"

CONT_COMMON_DATA="${CONT_COMMON_DATA:-/common_data}"
CONT_REPO="${CONT_REPO:-/code/medgemma}"

MODEL_ID="${MODEL_ID:-google/medgemma-1.5-4b-it}"
HF_HOME_IN_CONTAINER="${HF_HOME_IN_CONTAINER:-${CONT_COMMON_DATA}/medgemma}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

echo "[INFO] HOST_COMMON_DATA=${HOST_COMMON_DATA}"
echo "[INFO] HOST_REPO=${HOST_REPO}"
echo "[INFO] IMG=${IMG}"
echo "[INFO] HOST=${HOST} PORT=${PORT} DTYPE=${DTYPE} DEVICE_MAP=${DEVICE_MAP}"
echo "[INFO] HF_HOME_IN_CONTAINER=${HF_HOME_IN_CONTAINER}"

if [ ! -f "${IMG}" ]; then
  echo "[ERROR] Apptainer image not found: ${IMG}"
  echo "[HINT] Recreate it by pulling a vLLM OpenAI server image, e.g.:"
  echo "  apptainer pull --force \"${IMG}\" docker://vllm/vllm-openai:latest"
  exit 2
fi

# If MODEL_ID is an absolute HOST path under HOST_COMMON_DATA or HOST_REPO,
# translate it into the container mount path so Transformers can see it.
MODEL_ID_IN_CONTAINER="${MODEL_ID}"
if [[ "${MODEL_ID_IN_CONTAINER}" == "${HOST_COMMON_DATA}/"* ]]; then
  MODEL_ID_IN_CONTAINER="${CONT_COMMON_DATA}/${MODEL_ID_IN_CONTAINER#${HOST_COMMON_DATA}/}"
fi
if [[ "${MODEL_ID_IN_CONTAINER}" == "${HOST_REPO}/"* ]]; then
  MODEL_ID_IN_CONTAINER="${CONT_REPO}/${MODEL_ID_IN_CONTAINER#${HOST_REPO}/}"
fi
echo "[INFO] MODEL_ID(host)=${MODEL_ID}"
echo "[INFO] MODEL_ID(container)=${MODEL_ID_IN_CONTAINER}"

# Resolve script path for both layouts:
# 1) repo_root/MRI_Agent/scripts (nested layout)
# 2) repo_root/scripts (repo == MRI_Agent)
if [ -f "${HOST_REPO}/MRI_Agent/scripts/medgemma_openai_server.py" ]; then
  SERVER_SCRIPT="${CONT_REPO}/MRI_Agent/scripts/medgemma_openai_server.py"
elif [ -f "${HOST_REPO}/scripts/medgemma_openai_server.py" ]; then
  SERVER_SCRIPT="${CONT_REPO}/scripts/medgemma_openai_server.py"
else
  echo "[ERROR] medgemma_openai_server.py not found under HOST_REPO: ${HOST_REPO}"
  exit 2
fi
echo "[INFO] SERVER_SCRIPT(container)=${SERVER_SCRIPT}"

# Note: This server requires these python deps inside the container:
#   fastapi uvicorn pillow transformers accelerate (and torch).
# If your ${IMG} doesn't include them, install/build a new image or pip install inside.

apptainer exec --nv \
  -B "${HOST_COMMON_DATA}:${CONT_COMMON_DATA}" \
  -B "${HOST_REPO}:${CONT_REPO}" \
  --env HF_HOME="${HF_HOME_IN_CONTAINER}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "${IMG}" \
  python3 "${SERVER_SCRIPT}" \
    --model "${MODEL_ID_IN_CONTAINER}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"
