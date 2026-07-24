#!/usr/bin/env bash
set -euo pipefail

# Minimal parallel example for the BCER benchmark runner.
# Loops a couple of tasks against the `bcer` arm (fault=none) using the
# repaired benchmark/benchmark_runner.py CLI.
#
# Usage:
#   bash run_example_parallel.sh
#
# Optional env vars:
#   PARALLEL_JOBS=2                             # concurrent runs
#   MANIFEST=benchmark/cases_manifest.jsonl     # cases manifest (jsonl)
#   TASKS_REGISTRY=configs/tasks_registry.json  # task definitions
#   RUNS_ROOT=runs                              # per-run artifact/log root
#   RESULTS_DIR=benchmark                       # summary + log destination
#   SKIP_EXISTING=1                             # skip combos already scored
#   PYTHON_BIN=python

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Concurrency, defaulting to 2 or overridden via the environment.
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
MANIFEST="${MANIFEST:-benchmark/cases_manifest.jsonl}"
TASKS_REGISTRY="${TASKS_REGISTRY:-configs/tasks_registry.json}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
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

# Single paper arm: BCER (constrained sketch planner + Cerebellum + reflector).
ARMS=(
  bcer
)

# A couple of example tasks defined in the tasks registry.
TASKS=(
  short_superres
  long_cardiac_full
)

mkdir -p "${RESULTS_DIR}" "${RESULTS_DIR}/logs/example_parallel"
FAIL_FILE="${RESULTS_DIR}/example_parallel_failed_bcer.tsv"
: > "${FAIL_FILE}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[parallel-example] python not found: ${PYTHON_BIN}"
  exit 1
fi

PYTHON_EXE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

export MANIFEST TASKS_REGISTRY RUNS_ROOT RESULTS_DIR SKIP_EXISTING MAX_CASES
export MAX_NEW_TOKENS MAX_STEPS MAX_RETRIES CLEANUP_RUNS
export SERVER_BASE_URL SERVER_MODEL FAIL_FILE
export PYTHON_BIN

TOTAL=$(( ${#ARMS[@]} * ${#TASKS[@]} ))
echo "[parallel-example] total_combos=${TOTAL} jobs=${PARALLEL_JOBS}"
echo "[parallel-example] server=${SERVER_MODEL}@${SERVER_BASE_URL}"
echo "[parallel-example] manifest=${MANIFEST}"
echo "[parallel-example] python=${PYTHON_EXE}"

run_one() {
  local arm="$1"
  local task="$2"
  local out_json="${RESULTS_DIR}/benchmark_results_${task}_${arm}_none.json"
  local log_file="${RESULTS_DIR}/logs/example_parallel/${task}_${arm}_none.log"

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
      --runs-root "${RUNS_ROOT}" \
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

# Export the function so xargs-spawned shells can call it.
export -f run_one

# Fan out the arm x task combos across PARALLEL_JOBS workers.
{
  for arm in "${ARMS[@]}"; do
    for task in "${TASKS[@]}"; do
      printf "%s %s\n" "${arm}" "${task}"
    done
  done
} | xargs -P "${PARALLEL_JOBS}" -n 2 bash -c 'run_one "$@"' _

if [[ -s "${FAIL_FILE}" ]]; then
  echo
  echo "[parallel-example] done with failures. see: ${FAIL_FILE}"
  exit 1
fi

echo
echo "[parallel-example] done. all combos finished."
