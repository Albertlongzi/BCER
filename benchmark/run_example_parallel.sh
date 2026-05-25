#!/usr/bin/env bash
set -euo pipefail

# Parallel runner for bcer_sketch baseline (fault=none).
# Combos: 1 arm x 8 tasks = 8 runs.
#
# Usage:
#   bash run_bcer_sketch_parallel.sh
#
# Optional env vars:
#   PARALLEL_JOBS=6
#   MANIFEST=benchmark/cases_manifest_4.jsonl
#   TASKS_REGISTRY=configs/tasks_registry.json
#   RESULTS_DIR=benchmark
#   SKIP_EXISTING=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# 默认设置并发数为 8，或者通过环境变量覆盖
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
MANIFEST="${MANIFEST:-benchmark/cases_manifest_36.jsonl}"
TASKS_REGISTRY="${TASKS_REGISTRY:-configs/tasks_registry.json}"
RESULTS_DIR="${RESULTS_DIR:-benchmark}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_CASES="${MAX_CASES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_STEPS="${MAX_STEPS:-12}"
MAX_RETRIES="${MAX_RETRIES:-2}"
CLEANUP_RUNS="${CLEANUP_RUNS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SERVER_BASE_URL="${SERVER_BASE_URL:-${MRI_AGENT_SHELL_SERVER_BASE_URL:-http://127.0.0.1:8000/v1}}"
SERVER_MODEL="${SERVER_MODEL:-${MEDGEMMA_SERVER_MODEL:-Qwen/Qwen3-VL-30B-A3B-Thinking}}"

# 只跑这一个 arm
ARMS=(
  bcer_sketch
)

# 你提供的 8 个 tasks
TASKS=(
  short_superres
  long_cardiac_full
)

mkdir -p "${RESULTS_DIR}" "${RESULTS_DIR}/logs/baseline_parallel"
FAIL_FILE="${RESULTS_DIR}/baseline_parallel_failed_bcer_sketch.tsv"
: > "${FAIL_FILE}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[parallel-baseline] python not found: ${PYTHON_BIN}"
  exit 1
fi

PYTHON_EXE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

export MANIFEST TASKS_REGISTRY RESULTS_DIR SKIP_EXISTING MAX_CASES MAX_NEW_TOKENS MAX_STEPS MAX_RETRIES CLEANUP_RUNS
export SERVER_BASE_URL SERVER_MODEL FAIL_FILE
export PYTHON_BIN

TOTAL=$(( ${#ARMS[@]} * ${#TASKS[@]} ))
echo "[parallel-baseline] total_combos=${TOTAL} jobs=${PARALLEL_JOBS}"
echo "[parallel-baseline] server=${SERVER_MODEL}@${SERVER_BASE_URL}"
echo "[parallel-baseline] manifest=${MANIFEST}"
echo "[parallel-baseline] python=${PYTHON_EXE}"

# Grappa fallback 检查
if printf '%s\n' "${TASKS[@]}" | grep -q '^short_recon_grappa$'; then
  if ! "${PYTHON_BIN}" -c 'import pygrappa' >/dev/null 2>&1; then
    echo "[parallel-baseline] WARN: pygrappa import failed for ${PYTHON_EXE}; short_recon_grappa may fallback to image passthrough."
  fi
fi

run_one() {
  local arm="$1"
  local task="$2"
  local out_json="${RESULTS_DIR}/benchmark_results_v2_${task}_${arm}_none.json"
  local log_file="${RESULTS_DIR}/logs/baseline_parallel/${task}_${arm}_none.log"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${out_json}" ]]; then
    echo "[skip] arm=${arm} task=${task} (exists)"
    return 0
  fi

  echo "[run] arm=${arm} task=${task}"
  if "${PYTHON_BIN}" benchmark/benchmark_runner.py \
      --manifest "${MANIFEST}" \
      --task "${task}" \
      --arm "${arm}" \
      --fault none \
      --tasks-registry "${TASKS_REGISTRY}" \
      --server-base-url "${SERVER_BASE_URL}" \
      --server-model "${SERVER_MODEL}" \
      --max-cases "${MAX_CASES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-steps "${MAX_STEPS}" \
      --max-retries "${MAX_RETRIES}" \
      --cleanup-runs "${CLEANUP_RUNS}" \
      --output "${out_json}" \
      > "${log_file}" 2>&1; then
    echo "[ok] arm=${arm} task=${task}"
  else
    echo "[fail] arm=${arm} task=${task} (see ${log_file})"
    printf "%s\t%s\t%s\n" "${task}" "${arm}" "${log_file}" >> "${FAIL_FILE}"
  fi
}

# 导出函数供 xargs 调用
export -f run_one

# 并行执行
{
  for arm in "${ARMS[@]}"; do
    for task in "${TASKS[@]}"; do
      printf "%s %s\n" "${arm}" "${task}"
    done
  done
} | xargs -P "${PARALLEL_JOBS}" -n 2 bash -c 'run_one "$@"' _

if [[ -s "${FAIL_FILE}" ]]; then
  echo
  echo "[parallel-baseline] done with failures. see: ${FAIL_FILE}"
  exit 1
fi

echo
echo "[parallel-baseline] done. all combos finished."