#!/usr/bin/env bash
#
# build_demo_cases.sh — populate the three demo placeholder case dirs from a
# local open-dataset data root, in the exact layout the BCER tools +
# scripts/manifest_builder.py expect.
#
# NO medical data is ever committed to the repo: .gitignore excludes
# *.nii/*.nii.gz/*.dcm/*.h5, so the files this script stages under
# demo/cases/<case>/ stay untracked. Re-running is safe (idempotent): each
# case dir is cleared (except its .gitkeep) and re-staged.
#
# Datasets used (all public; you must obtain them yourself — see demo/README.md
# for provenance, licensing and registration):
#   - brain    : BraTS 2018 4-modality NIfTI  -> demo/cases/Brats18_CBICA_AAM_1
#   - prostate : fastMRI Prostate DICOM series -> demo/cases/sub-019_2
#   - cardiac  : ACDC cine NIfTI (single frame) -> demo/cases/acdc_multiseq_patient061_ed
#
# There are no built-in dataset paths: you must say where your local copies
# live. DATA_ROOT covers brain + prostate; cardiac (ACDC) is pointed at
# separately because it usually lives in its own tree.
#
# Usage:
#   DATA_ROOT=/my/data demo/build_demo_cases.sh
#   demo/build_demo_cases.sh --data-root /my/data
#   CARDIAC_SRC=/path/to/acdc/TaskDir demo/build_demo_cases.sh --data-root /my/data
#   BRAIN_SRC=... PROSTATE_SRC=... CARDIAC_SRC=... demo/build_demo_cases.sh
#   demo/build_demo_cases.sh --data-root /my/data --skip-brain   # subset only
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the demo dir (this script lives in demo/).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DEMO_DIR="${SCRIPT_DIR}"
CASES_DIR="${DEMO_DIR}/cases"

# ---------------------------------------------------------------------------
# Configuration (env-overridable; CLI flags below override env).
# ---------------------------------------------------------------------------
# No default: point this at your own copy of the open datasets.
DATA_ROOT="${DATA_ROOT:-}"

SKIP_BRAIN=0
SKIP_PROSTATE=0
SKIP_CARDIAC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)     DATA_ROOT="$2"; shift 2 ;;
    --data-root=*)   DATA_ROOT="${1#*=}"; shift ;;
    --skip-brain)    SKIP_BRAIN=1; shift ;;
    --skip-prostate) SKIP_PROSTATE=1; shift ;;
    --skip-cardiac)  SKIP_CARDIAC=1; shift ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^#//'; exit 0 ;;
    *) echo "[ERR] unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Per-domain source paths. DATA_ROOT supplies the brain + prostate defaults;
# the ACDC cine tree is named separately via CARDIAC_SRC. Any of the three can
# be overridden individually, in which case DATA_ROOT is not needed for it.
BRAIN_SRC="${BRAIN_SRC:-${DATA_ROOT:+${DATA_ROOT}/Brats2018/MICCAI_BraTS_2018_Data_Validation/Brats18_CBICA_AAM_1}}"
PROSTATE_SRC="${PROSTATE_SRC:-${DATA_ROOT:+${DATA_ROOT}/fastmri_prostate_v3/subjects/sub-019/DICOMS}}"
CARDIAC_SRC="${CARDIAC_SRC:-}"

# Fail loudly rather than probing paths the user never named.
_missing_src=()
[[ "${SKIP_BRAIN}"    -eq 1 || -n "${BRAIN_SRC}"    ]] || _missing_src+=("brain (set DATA_ROOT or BRAIN_SRC)")
[[ "${SKIP_PROSTATE}" -eq 1 || -n "${PROSTATE_SRC}" ]] || _missing_src+=("prostate (set DATA_ROOT or PROSTATE_SRC)")
[[ "${SKIP_CARDIAC}"  -eq 1 || -n "${CARDIAC_SRC}"  ]] || _missing_src+=("cardiac (set CARDIAC_SRC=/path/to/acdc/TaskDir)")
if [[ "${#_missing_src[@]}" -gt 0 ]]; then
  echo "[demo][ERR] no data source given for: ${_missing_src[*]}" >&2
  echo "[demo][ERR] see demo/README.md for the expected dataset layouts, or pass --skip-<domain>." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '[demo] %s\n' "$*"; }
warn() { printf '[demo][WARN] %s\n' "$*" >&2; }

# Remove everything in a case dir except its .gitkeep placeholder.
clean_case_dir() {
  local dir="$1"
  mkdir -p "$dir"
  find "$dir" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# BRAIN — BraTS 2018 validation case -> demo/cases/Brats18_CBICA_AAM_1
#
# Expected layout (BraTS style, one dir per case; manifest builder + brain tools
# detect modalities from the t1/t1ce/t2/flair filename tokens):
#   Brats18_CBICA_AAM_1/
#     Brats18_CBICA_AAM_1_t1.nii.gz     (T1)
#     Brats18_CBICA_AAM_1_t1ce.nii.gz   (T1c)
#     Brats18_CBICA_AAM_1_t2.nii.gz     (T2)
#     Brats18_CBICA_AAM_1_flair.nii.gz  (FLAIR)
#
# Source files are plain .nii; identify_sequences accepts both .nii and
# .nii.gz, but we gzip to match the canonical BraTS .nii.gz layout in
# docs/DATASETS.md. gzip is CPU-only (no GPU) but not run here — this stages it.
# ---------------------------------------------------------------------------
stage_brain() {
  local dst="${CASES_DIR}/Brats18_CBICA_AAM_1"
  log "brain: source = ${BRAIN_SRC}"
  if [[ ! -d "${BRAIN_SRC}" ]]; then
    warn "brain source dir not found: ${BRAIN_SRC} — skipping brain."
    return 0
  fi
  clean_case_dir "${dst}"
  local staged=0
  local mod f base out
  for mod in t1 t1ce t2 flair; do
    # match either <case>_<mod>.nii or <case>_<mod>.nii.gz
    f="$(ls "${BRAIN_SRC}"/*"_${mod}".nii "${BRAIN_SRC}"/*"_${mod}".nii.gz 2>/dev/null | head -n1 || true)"
    if [[ -z "${f}" ]]; then
      warn "brain: modality '${mod}' not found under ${BRAIN_SRC}"
      continue
    fi
    base="Brats18_CBICA_AAM_1_${mod}.nii.gz"
    out="${dst}/${base}"
    if [[ "${f}" == *.nii.gz ]]; then
      cp -f "${f}" "${out}"
    else
      # gzip .nii -> .nii.gz (CPU-only)
      gzip -c "${f}" > "${out}"
    fi
    log "brain: staged ${base}"
    staged=$((staged+1))
  done
  log "brain: staged ${staged}/4 modalities -> ${dst}"
}

# ---------------------------------------------------------------------------
# PROSTATE — fastMRI Prostate sub-019 -> demo/cases/sub-019_2
#
# Expected layout (DICOM series subfolders; identify_sequences enumerates
# immediate series subdirs of the case dir and converts each to NIfTI):
#   sub-019_2/
#     AX_T2/                 *.dcm   (T2w reference)
#     AX_DIFFUSION_ADC/      *.dcm   (ADC)
#     AX_DIFFUSION_TRACEW/   *.dcm   (DWI / high-b trace)
#     AX_DIFFUSION_CALC_BVAL/*.dcm   (calculated high-b)
# ---------------------------------------------------------------------------
stage_prostate() {
  local dst="${CASES_DIR}/sub-019_2"
  log "prostate: source = ${PROSTATE_SRC}"
  if [[ ! -d "${PROSTATE_SRC}" ]]; then
    warn "prostate source dir not found: ${PROSTATE_SRC} — skipping prostate."
    return 0
  fi
  clean_case_dir "${dst}"
  local staged=0 series
  for series in AX_T2 AX_DIFFUSION_ADC AX_DIFFUSION_TRACEW AX_DIFFUSION_CALC_BVAL; do
    if [[ -d "${PROSTATE_SRC}/${series}" ]]; then
      cp -r "${PROSTATE_SRC}/${series}" "${dst}/${series}"
      log "prostate: staged series ${series}/"
      staged=$((staged+1))
    else
      warn "prostate: series '${series}' not found under ${PROSTATE_SRC}"
    fi
  done
  log "prostate: staged ${staged} DICOM series -> ${dst}"
}

# ---------------------------------------------------------------------------
# CARDIAC — ACDC patient061 ED -> demo/cases/acdc_multiseq_patient061_ed
#
# Expected layout (cine NIfTI; identify_sequences maps patientNNN_frameNN* to
# the CINE sequence and excludes *_gt volumes from sequence identification):
#   acdc_multiseq_patient061_ed/
#     patient061_frame01_0000.nii.gz   (ED single-frame 3D cine — the CINE input)
#     patient061_frame01_gt.nii.gz     (ED ground-truth seg; *_gt excluded from identify)
#
# NOTE: this is a SINGLE ED frame (3D), not the full 4D cine. The long_cardiac_full
# chain expects a 4D cine (patient061_4d.nii.gz) which is NOT on local disk.
# See demo/README.md "Cardiac 4D-cine gap" for the official ACDC download.
# Only the ED frame is staged as CINE to keep sequence resolution unambiguous;
# the ES frame (frame10) is available at the source if you want to add it.
# ---------------------------------------------------------------------------
stage_cardiac() {
  local dst="${CASES_DIR}/acdc_multiseq_patient061_ed"
  local img_dir="${CARDIAC_SRC}/imagesTr_single"
  local lbl_dir="${CARDIAC_SRC}/labelsTr"
  log "cardiac: source = ${CARDIAC_SRC}"
  if [[ ! -d "${img_dir}" ]]; then
    warn "cardiac image dir not found: ${img_dir} — skipping cardiac."
    return 0
  fi
  clean_case_dir "${dst}"
  local staged=0
  local ed="${img_dir}/patient061_frame01_0000.nii.gz"
  if [[ -f "${ed}" ]]; then
    cp -f "${ed}" "${dst}/patient061_frame01_0000.nii.gz"
    log "cardiac: staged ED cine patient061_frame01_0000.nii.gz"
    staged=$((staged+1))
  else
    warn "cardiac: ED frame not found: ${ed}"
  fi
  local gt="${lbl_dir}/patient061_frame01.nii.gz"
  if [[ -f "${gt}" ]]; then
    # rename to *_gt so identify_sequences excludes it from sequence detection
    cp -f "${gt}" "${dst}/patient061_frame01_gt.nii.gz"
    log "cardiac: staged GT seg patient061_frame01_gt.nii.gz (excluded from identify)"
    staged=$((staged+1))
  else
    warn "cardiac: GT seg not found: ${gt} (optional)"
  fi
  log "cardiac: staged ${staged} file(s) -> ${dst}"
  warn "cardiac uses a SINGLE ED frame (3D), not 4D cine — see demo/README.md."
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
log "DATA_ROOT = ${DATA_ROOT}"
log "cases dir = ${CASES_DIR}"

[[ "${SKIP_BRAIN}"    -eq 1 ]] || stage_brain
[[ "${SKIP_PROSTATE}" -eq 1 ]] || stage_prostate
[[ "${SKIP_CARDIAC}"  -eq 1 ]] || stage_cardiac

log "done. Staged data is untracked (see .gitignore). Build the manifest next:"
log "  demo/run_longest_chain.sh --build-manifest-only"
