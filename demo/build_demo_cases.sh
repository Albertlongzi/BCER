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
# Which ACDC patient to stage. The tracked placeholder dir under demo/cases/ is
# named for patient061; changing this stages into a sibling dir instead.
CARDIAC_PATIENT="${CARDIAC_PATIENT:-patient061}"

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
# CARDIAC — one ACDC patient -> demo/cases/acdc_multiseq_<patient>_ed
#
# CARDIAC_SRC may point at any of three layouts:
#   A. a native ACDC patient dir            .../database/training/patient061/
#   B. a native ACDC split root             .../database/training/   (descends to
#                                            ${CARDIAC_PATIENT})
#   C. an nnUNet-converted task dir         .../TaskNNN_*/imagesTr_single + labelsTr
#
# Layouts A and B are what you get from the official ACDC download; C is the
# preprocessed form. The full 4D cine is preferred when present, because
# long_cardiac_full can then derive ED/ES phases and ejection fractions. When
# only single phases exist, the ED frame alone is staged so that sequence
# resolution stays unambiguous (the chain still runs, but EF-style features
# degrade). Ground-truth volumes are staged as *_gt, which identify_sequences
# excludes from sequence detection.
# ---------------------------------------------------------------------------
stage_cardiac() {
  local patient="${CARDIAC_PATIENT}"
  local dst="${CASES_DIR}/acdc_multiseq_${patient}_ed"
  log "cardiac: source = ${CARDIAC_SRC} (patient=${patient})"

  # Resolve the layout.
  local img_dir="" lbl_dir="" layout=""
  if [[ -d "${CARDIAC_SRC}/imagesTr_single" ]]; then
    img_dir="${CARDIAC_SRC}/imagesTr_single"; lbl_dir="${CARDIAC_SRC}/labelsTr"; layout="nnunet"
  elif compgen -G "${CARDIAC_SRC}/${patient}_*.nii.gz" >/dev/null 2>&1; then
    img_dir="${CARDIAC_SRC}"; lbl_dir="${CARDIAC_SRC}"; layout="acdc-patient"
  elif [[ -d "${CARDIAC_SRC}/${patient}" ]]; then
    img_dir="${CARDIAC_SRC}/${patient}"; lbl_dir="${img_dir}"; layout="acdc-root"
  else
    warn "cardiac: no recognized ACDC layout under ${CARDIAC_SRC} — skipping cardiac."
    warn "cardiac: expected ${patient}_*.nii.gz, a ${patient}/ subdir, or imagesTr_single/."
    return 0
  fi
  log "cardiac: detected layout '${layout}' (images=${img_dir})"

  clean_case_dir "${dst}"
  local staged=0

  # Preferred input: the full 4D cine.
  local cine4d=""
  for cand in "${img_dir}/${patient}_4d.nii.gz" "${img_dir}/${patient}_4d_0000.nii.gz"; do
    [[ -f "${cand}" ]] && { cine4d="${cand}"; break; }
  done

  if [[ -n "${cine4d}" ]]; then
    cp -f "${cine4d}" "${dst}/${patient}_4d.nii.gz"
    log "cardiac: staged 4D cine ${patient}_4d.nii.gz (full cine — ED/ES derivable)"
    staged=$((staged+1))
  else
    # Fall back to the ED single phase, under either naming convention.
    local ed=""
    for cand in "${img_dir}/${patient}_frame01_0000.nii.gz" "${img_dir}/${patient}_frame01.nii.gz"; do
      [[ -f "${cand}" ]] && { ed="${cand}"; break; }
    done
    if [[ -n "${ed}" ]]; then
      cp -f "${ed}" "${dst}/$(basename "${ed}")"
      log "cardiac: staged ED cine $(basename "${ed}")"
      staged=$((staged+1))
    else
      warn "cardiac: neither ${patient}_4d.nii.gz nor ${patient}_frame01*.nii.gz found under ${img_dir}"
    fi
  fi

  # Ground truth for the ED phase (optional). In the nnUNet layout the label has
  # no _gt suffix, so add one; in the native layout it already has it.
  local gt=""
  for cand in "${lbl_dir}/${patient}_frame01_gt.nii.gz" "${lbl_dir}/${patient}_frame01.nii.gz"; do
    [[ -f "${cand}" && "${cand}" != "${img_dir}/${patient}_frame01.nii.gz" ]] && { gt="${cand}"; break; }
  done
  if [[ -n "${gt}" ]]; then
    cp -f "${gt}" "${dst}/${patient}_frame01_gt.nii.gz"
    log "cardiac: staged GT seg ${patient}_frame01_gt.nii.gz (excluded from identify)"
    staged=$((staged+1))
  else
    warn "cardiac: ED ground-truth seg not found under ${lbl_dir} (optional)"
  fi

  log "cardiac: staged ${staged} file(s) -> ${dst}"
  [[ -n "${cine4d}" ]] || warn "cardiac: single ED phase only (3D), not 4D cine — EF-style features degrade."
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
