#!/bin/bash
set -euo pipefail

# Start an OpenAI-compatible vLLM server inside apptainer.
# Run this on a GPU compute node.
#
# Example:
#   module load apptainer/1.0.1
#   bash MRI_Agent/scripts/start_vllm_server_apptainer.sh \
#     --tp 8 --port 8000 --max-model-len 32768

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST_REPO="${HOST_REPO:-${DEFAULT_REPO}}"
HOST_COMMON_DATA="${HOST_COMMON_DATA:-${HOST_REPO}}"
IMG="${IMG:-${HOST_COMMON_DATA}/vllm_latest.sif}"

CONT_COMMON_DATA="${CONT_COMMON_DATA:-/common_data}"
CONT_REPO="${CONT_REPO:-/code/medgemma}"

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${HOST_COMMON_DATA}/medgemma}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Thinking}"
HF_HOME_IN_CONTAINER="${HF_HOME_IN_CONTAINER:-${CONT_COMMON_DATA}/medgemma}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"

TP="${TP:-8}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# Offline knobs (default ON for HPC). If your local HF cache is missing/broken, set these to 0
# for a one-time online repair/download on a node with outbound network access.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Eager execution mode. 
# Default to 0 (OFF) for performance. Set to 1 (ON) only if debugging or OOM.
# CUDA Graphs (non-eager) provide significantly lower latency.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# Expert-parallel (MoE) knobs. This model is MoE and tends to work better with EP enabled.
# vLLM CLI supports:
#   --enable-expert-parallel / --no-enable-expert-parallel
#   --expert-placement-strategy {linear,round_robin}
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"   # 1=enable, 0=disable
EXPERT_PLACEMENT_STRATEGY="${EXPERT_PLACEMENT_STRATEGY:-round_robin}"

# Multimodal OpenAI message content format.
# For Qwen-VL style requests, we send messages where `content` is a LIST of blocks:
#   [{"type":"text",...},{"type":"image_url",...}]
# Some vLLM setups auto-detect as "string" and may ignore image blocks unless forced.
CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-openai}"  # openai|string

echo "[INFO] HOST_COMMON_DATA=${HOST_COMMON_DATA}"
echo "[INFO] HOST_REPO=${HOST_REPO}"
echo "[INFO] IMG=${IMG}"
echo "[INFO] MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT}"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[INFO] TP=${TP} PORT=${PORT} HOST=${HOST} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[INFO] ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL} EXPERT_PLACEMENT_STRATEGY=${EXPERT_PLACEMENT_STRATEGY}"
echo "[INFO] ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "[INFO] CHAT_TEMPLATE_CONTENT_FORMAT=${CHAT_TEMPLATE_CONTENT_FORMAT}"
echo "[INFO] HF_HOME_IN_CONTAINER=${HF_HOME_IN_CONTAINER} HF_HUB_OFFLINE=${HF_HUB_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"

if [ ! -f "${IMG}" ]; then
  echo "[ERROR] Apptainer image not found: ${IMG}"
  echo "[HINT] Recreate it by pulling a vLLM OpenAI server image, e.g.:"
  echo "  apptainer pull --force \"${IMG}\" docker://vllm/vllm-openai:latest"
  echo "[HINT] If your cluster blocks outbound network on compute nodes, run the pull on a login node (or any node with network), then re-run this script."
  exit 2
fi

# If MODEL_ID is an absolute HOST path under HOST_COMMON_DATA or HOST_REPO,
# translate it into the container mount path so vLLM can see it.
MODEL_ID_RESOLVED="${MODEL_ID}"
if [[ "${MODEL_ID_RESOLVED}" != /* ]]; then
  LOCAL_REPO="${MODEL_CACHE_ROOT}/hub/models--${MODEL_ID_RESOLVED//\//--}"
  if [ -f "${LOCAL_REPO}/refs/main" ]; then
    SNAP="$(cat "${LOCAL_REPO}/refs/main" | tr -d '\r\n')"
    CAND="${LOCAL_REPO}/snapshots/${SNAP}"
    if [ -d "${CAND}" ]; then
      MODEL_ID_RESOLVED="${CAND}"
    fi
  fi
fi

MODEL_ID_IN_CONTAINER="${MODEL_ID_RESOLVED}"
if [[ "${MODEL_ID_IN_CONTAINER}" == "${HOST_COMMON_DATA}/"* ]]; then
  MODEL_ID_IN_CONTAINER="${CONT_COMMON_DATA}/${MODEL_ID_IN_CONTAINER#${HOST_COMMON_DATA}/}"
fi
if [[ "${MODEL_ID_IN_CONTAINER}" == "${HOST_REPO}/"* ]]; then
  MODEL_ID_IN_CONTAINER="${CONT_REPO}/${MODEL_ID_IN_CONTAINER#${HOST_REPO}/}"
fi
echo "[INFO] MODEL_ID(host)=${MODEL_ID}"
echo "[INFO] MODEL_ID(resolved)=${MODEL_ID_RESOLVED}"
echo "[INFO] MODEL_ID(container)=${MODEL_ID_IN_CONTAINER}"
echo "[INFO] SERVED_MODEL_NAME(container)=${SERVED_MODEL_NAME}"

apptainer exec --nv \
  -B "${HOST_COMMON_DATA}:${CONT_COMMON_DATA}" \
  -B "${HOST_REPO}:${CONT_REPO}" \
  --env HF_HOME="${HF_HOME_IN_CONTAINER}" \
  --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  --env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
  "${IMG}" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID_IN_CONTAINER}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    $( [ "${ENABLE_EXPERT_PARALLEL}" = "1" ] && echo "--enable-expert-parallel" || echo "--no-enable-expert-parallel" ) \
    $( [ "${ENABLE_PREFIX_CACHING}" = "1" ] && echo "--enable-prefix-caching" || echo "" ) \
    $( [ "${ENFORCE_EAGER}" = "1" ] && echo "--enforce-eager" || echo "" ) \
    --expert-placement-strategy "${EXPERT_PLACEMENT_STRATEGY}" \
    --chat-template-content-format "${CHAT_TEMPLATE_CONTENT_FORMAT}" \
    --trust-remote-code \
    --mm-encoder-attn-backend TORCH_SDPA \
    --mm-encoder-tp-mode data \
    --disable-log-stats
