#!/usr/bin/env bash
#
# run_longest_chain.sh — demonstrate the LONGEST BCER tool chain for each of the
# three domains (brain, cardiac, prostate) over the demo cases.
#
# It (1) builds a benchmark manifest from the populated demo/cases dirs with
# scripts/manifest_builder.py, then (2) runs the `bcer` arm on the longest task
# contract for each domain via benchmark/benchmark_runner.py.
#
# PREREQUISITES
#   1. Populate the demo cases first:   demo/build_demo_cases.sh
#   2. For --execute, an OpenAI-compatible chat-completions endpoint must be
#      reachable at --server-base-url (vLLM / llama.cpp / SGLang all work; the
#      tools do the imaging work, the model only plans and writes the report).
#      Use --build-manifest-only to just build the manifest (needs no server).
#
# Longest chain per domain (see agent/plans/templates/*_full_pipeline.json):
#   prostate : long_prostate_full   identify -> register(ADC,DWI) -> segment
#                                    -> detect_lesion -> features -> package
#                                    -> report                        (registry task)
#   cardiac  : long_cardiac_full    identify -> segment_cine -> classify
#                                    -> features -> qa -> package -> report
#                                                                     (registry task)
#   brain    : long_brain_full      identify -> segment -> features
#                                    -> classify_grade -> package -> report
#                                                                     (registry task)
#              The matching template agent/plans/templates/brain_full_pipeline.json
#              also carries a T2/FLAIR->T1c registration node, which the planner
#              normally skips on BraTS data (already co-registered at 1x1x1 mm).
#              Set BRAIN_TASK=medium_brain_grade_classify for the shorter chain.
#
# Usage:
#   demo/run_longest_chain.sh                       # build manifest + print runner cmds
#   demo/run_longest_chain.sh --build-manifest-only # only build the manifest
#   demo/run_longest_chain.sh --execute             # actually invoke the runner
#                                                   # (requires a live inference server)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DEMO_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${DEMO_DIR}/.." >/dev/null 2>&1 && pwd)"
CASES_DIR="${DEMO_DIR}/cases"

MANIFEST="${MANIFEST:-${DEMO_DIR}/cases_manifest.jsonl}"
TASKS_REGISTRY="${TASKS_REGISTRY:-${REPO_ROOT}/configs/tasks_registry.json}"
ARM="${ARM:-bcer}"
FAULT="${FAULT:-none}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"

# Per-domain demo case roots (populated by build_demo_cases.sh).
BRAIN_CASE="${CASES_DIR}/Brats18_CBICA_AAM_1"
PROSTATE_CASE="${CASES_DIR}/sub-019_2"
CARDIAC_CASE="${CASES_DIR}/acdc_multiseq_patient061_ed"

# Longest task id per domain.
PROSTATE_TASK="${PROSTATE_TASK:-long_prostate_full}"
CARDIAC_TASK="${CARDIAC_TASK:-long_cardiac_full}"
BRAIN_TASK="${BRAIN_TASK:-long_brain_full}"

BUILD_ONLY=0
EXECUTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-manifest-only) BUILD_ONLY=1; shift ;;
    --execute)             EXECUTE=1; shift ;;
    -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^#//'; exit 0 ;;
    *) echo "[ERR] unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[demo] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Build the manifest from the demo case dirs.
#
# The manifest builder scans each --*-root as a "case root" (immediate
# subdirs are cases). The demo case dirs are themselves single cases, so we
# pass the parent (cases/) via per-domain roots; the builder's single-case
# fallback + one-level nesting handles either shape. Passing the individual
# case dirs directly via generic --case-root keeps modality inference scoped
# to exactly the demo case.
# ---------------------------------------------------------------------------
build_manifest() {
  log "building manifest -> ${MANIFEST}"
  local args=(--tasks-registry "${TASKS_REGISTRY}" --output "${MANIFEST}")
  [[ -d "${BRAIN_CASE}"    ]] && args+=(--brain-root    "${BRAIN_CASE}")    || log "skip brain: ${BRAIN_CASE} missing (run build_demo_cases.sh)"
  [[ -d "${CARDIAC_CASE}"  ]] && args+=(--cardiac-root  "${CARDIAC_CASE}")  || log "skip cardiac: ${CARDIAC_CASE} missing"

  # The brain and cardiac demo dirs contain only files, so the builder's
  # single-case fallback treats each as one case. The prostate case is a DICOM
  # *study* whose subdirs are SERIES (AX_T2, AX_DIFFUSION_ADC, ...). Pointing
  # the builder straight at it would emit four series-level cases, none of
  # which carries every modality long_prostate_full needs -- the runner would
  # then exit with "No cases selected from manifest". Hand it a scratch root
  # containing just a symlink to the study so the study stays a single case.
  # case_root in the output is resolved to the real path, so the scratch dir is
  # safe to delete afterwards.
  local prostate_link_root=""
  if [[ -d "${PROSTATE_CASE}" ]]; then
    prostate_link_root="$(mktemp -d)"
    ln -s "${PROSTATE_CASE}" "${prostate_link_root}/$(basename "${PROSTATE_CASE}")"
    args+=(--prostate-root "${prostate_link_root}")
  else
    log "skip prostate: ${PROSTATE_CASE} missing"
  fi

  ( cd "${REPO_ROOT}" && python scripts/manifest_builder.py "${args[@]}" )
  [[ -n "${prostate_link_root}" ]] && rm -rf "${prostate_link_root}"
  log "manifest built. Contents:"
  cat "${MANIFEST}" || true
}

# ---------------------------------------------------------------------------
# 2. Print (or run) the longest-chain bcer command for each domain.
# ---------------------------------------------------------------------------
runner_cmd() {
  local task="$1"
  cat <<EOF
python benchmark/benchmark_runner.py \\
    --manifest ${MANIFEST} \\
    --task ${task} \\
    --arm ${ARM} \\
    --fault ${FAULT} \\
    --runs-root ${RUNS_ROOT}
EOF
}

run_or_print() {
  local domain="$1" task="$2"
  log "----- ${domain}: longest chain -> task '${task}', arm '${ARM}' -----"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    ( cd "${REPO_ROOT}" && \
      python benchmark/benchmark_runner.py \
        --manifest "${MANIFEST}" \
        --task "${task}" \
        --arm "${ARM}" \
        --fault "${FAULT}" \
        --runs-root "${RUNS_ROOT}" )
  else
    runner_cmd "${task}"
  fi
}

# ---------------------------------------------------------------------------
main() {
  build_manifest
  [[ "${BUILD_ONLY}" -eq 1 ]] && { log "manifest-only: done."; return 0; }

  if [[ "${EXECUTE}" -eq 0 ]]; then
    log "DRY RUN: printing the intended runner commands (requires the repaired"
    log "benchmark runner + a live inference server). Re-run with --execute to run."
  fi

  run_or_print "prostate" "${PROSTATE_TASK}"
  run_or_print "cardiac"  "${CARDIAC_TASK}"
  run_or_print "brain"    "${BRAIN_TASK}"

  log "done."
}

main
