#!/bin/bash
# =============================================================================
# download_assets.sh -- one-command model-weights setup for BCER_open
# =============================================================================
# Populates the project-local assets/ tree that the tools expect:
#
#   assets/models/prostate_mri_anatomy/          MONAI bundle  (models/model.ts)
#   assets/models/brats_mri_segmentation/        MONAI bundle
#   assets/models/prostate_mri_lesion_seg/       app + weights (NON-COMMERCIAL)
#   assets/models/cardiac_nnunet/results/        nnUNet RESULTS_FOLDER
#   assets/checkpoints/                          (prostate_distortion REMOVED)
#
# The script is idempotent: a model that is already present is skipped unless
# --force is given. It never fails the whole run because of one optional model
# (cardiac in particular is left "soft" -- see the cardiac section).
#
# See docs/ASSETS.md for the full source/URL/license table.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- target layout ----------------------------------------------------------
ASSETS_DIR="${REPO_ROOT}/assets"
MODELS_DIR="${ASSETS_DIR}/models"
CHECKPOINTS_DIR="${ASSETS_DIR}/checkpoints"

ANATOMY_DIR="${MODELS_DIR}/prostate_mri_anatomy"
BRATS_DIR="${MODELS_DIR}/brats_mri_segmentation"
LESION_DIR="${MODELS_DIR}/prostate_mri_lesion_seg"
LESION_APP_DIR="${LESION_DIR}/prostate_mri_lesion_seg_app"
LESION_WEIGHT_DIR="${LESION_DIR}/weight"
CARDIAC_DIR="${MODELS_DIR}/cardiac_nnunet"
CARDIAC_RESULTS_DIR="${CARDIAC_DIR}/results"

# ---- optional pre-existing local copies -------------------------------------
# Only used by --from-local, and only if you point them somewhere. There are no
# built-in paths: MEDGEMMA_ROOT should hold models/prostate_mri_anatomy and
# prostate_mri_lesion_seg/{weight,scripts/...}; CMR_REVERSE_ROOT should hold
# download_nnunet_weights.sh. Everything works without them over the network.
MEDGEMMA_ROOT="${MEDGEMMA_ROOT:-}"
CMR_REVERSE_ROOT="${CMR_REVERSE_ROOT:-}"

LOCAL_ANATOMY="${MEDGEMMA_ROOT:+${MEDGEMMA_ROOT}/models/prostate_mri_anatomy}"
LOCAL_LESION_APP="${MEDGEMMA_ROOT:+${MEDGEMMA_ROOT}/prostate_mri_lesion_seg/scripts/research-contributions/prostate-mri-lesion-seg/prostate_mri_lesion_seg_app}"
LOCAL_LESION_WEIGHT="${MEDGEMMA_ROOT:+${MEDGEMMA_ROOT}/prostate_mri_lesion_seg/weight}"

# ---- remote sources ---------------------------------------------------------
LESION_GDRIVE_FOLDER="https://drive.google.com/drive/folders/1EpjrlzEdV7CcaCYqGTIEzOapamP4Ag6M"
RESEARCH_CONTRIB_REPO="https://github.com/Project-MONAI/research-contributions.git"
TASK027_URL="https://zenodo.org/records/3734294/files/Task027_ACDC.zip?download=1"

# ---- defaults / flags -------------------------------------------------------
ONLY=""                       # comma list: prostate,brats,lesion,cardiac (empty = all)
FROM_LOCAL=0                  # prefer local copies under $MEDGEMMA_ROOT
USE_SYMLINK=0                 # symlink instead of copy in --from-local mode
FORCE=0                       # re-fetch even if target already populated
CARDIAC_SOURCE="${CARDIAC_SOURCE:-auto}"  # auto | task027 | task900

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
info()  { printf '%s[INFO]%s %s\n' "${BLU}" "${RST}" "$*"; }
ok()    { printf '%s[ OK ]%s %s\n' "${GRN}" "${RST}" "$*"; }
warn()  { printf '%s[WARN]%s %s\n' "${YLW}" "${RST}" "$*" >&2; }
err()   { printf '%s[FAIL]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

usage() {
  cat <<EOF
Usage: $0 [options]

One-command setup of model weights into ${ASSETS_DIR}.

Options:
  --only LIST      Comma-separated subset to fetch. Members:
                     prostate  (alias: anatomy)  -> prostate_mri_anatomy
                     brats                        -> brats_mri_segmentation
                     lesion                       -> prostate_mri_lesion_seg
                     cardiac                      -> cardiac_nnunet
                   Default: all of the above.
  --from-local     Reuse copies you already have under \$MEDGEMMA_ROOT instead of
                   downloading; falls back to the network for anything missing.
                   Requires MEDGEMMA_ROOT to be set (there is no default).
  --symlink        In --from-local mode, symlink instead of copying (saves disk).
  --force          Re-fetch even if the target already looks populated.
  -h, --help       Show this help.

Environment (all optional; no built-in paths):
  MEDGEMMA_ROOT        Root of existing local copies, used only by --from-local.
                       Expected inside it: models/prostate_mri_anatomy and
                       prostate_mri_lesion_seg/{weight,scripts/research-contributions/...}
  CMR_REVERSE_ROOT     Directory containing download_nnunet_weights.sh, for the
                       collaborator cardiac Task900 weights (CARDIAC_SOURCE=task900).
  CARDIAC_SOURCE       auto (default; print options only) | task027 | task900

Examples:
  $0                              # download everything from the network
  $0 --only prostate,brats        # just the two MONAI bundles
  MEDGEMMA_ROOT=/my/models $0 --from-local --symlink   # reuse local copies
  CARDIAC_SOURCE=task027 $0 --only cardiac   # fetch public ACDC fallback

NOTE: prostate_mri_lesion_seg and the cardiac ACDC weights are NON-COMMERCIAL /
research-use-only. See docs/ASSETS.md LICENSE NOTES before use.
EOF
}

# ---- arg parsing ------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --only)        ONLY="${2:-}"; shift 2 ;;
    --only=*)      ONLY="${1#*=}"; shift ;;
    --from-local)  FROM_LOCAL=1; shift ;;
    --medgemma-root)   MEDGEMMA_ROOT="${2:-}"; shift 2 ;;
    --medgemma-root=*) MEDGEMMA_ROOT="${1#*=}"; shift ;;
    --symlink)     USE_SYMLINK=1; shift ;;
    --force)       FORCE=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             err "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# Re-derive the local copy paths now that --medgemma-root may have changed the root.
LOCAL_ANATOMY="${MEDGEMMA_ROOT:+${MEDGEMMA_ROOT}/models/prostate_mri_anatomy}"
LOCAL_LESION_APP="${MEDGEMMA_ROOT:+${MEDGEMMA_ROOT}/prostate_mri_lesion_seg/scripts/research-contributions/prostate-mri-lesion-seg/prostate_mri_lesion_seg_app}"
LOCAL_LESION_WEIGHT="${MEDGEMMA_ROOT:+${MEDGEMMA_ROOT}/prostate_mri_lesion_seg/weight}"

if [ "${FROM_LOCAL}" -eq 1 ] && [ -z "${MEDGEMMA_ROOT}" ]; then
  warn "--from-local given but MEDGEMMA_ROOT is not set; nothing can be reused locally."
  warn "Set MEDGEMMA_ROOT=/path/to/existing/models (or pass --medgemma-root). Falling back to network downloads."
  FROM_LOCAL=0
fi

want() {
  # want <name> -> 0 if selected (ONLY empty means everything)
  local name="$1"
  [ -z "${ONLY}" ] && return 0
  local IFS=','
  for m in ${ONLY}; do
    m="$(echo "${m}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    case "${m}" in
      anatomy) m="prostate" ;;
    esac
    [ "${m}" = "${name}" ] && return 0
  done
  return 1
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# place_local <src> <dst>  -- copy or symlink a local tree
place_local() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  rm -rf "${dst}"
  if [ "${USE_SYMLINK}" -eq 1 ]; then
    ln -s "${src}" "${dst}"
    info "symlinked ${dst} -> ${src}"
  else
    cp -a "${src}" "${dst}"
    info "copied ${src} -> ${dst}"
  fi
}

# ---- summary bookkeeping ----------------------------------------------------
declare -a SUMMARY

record() { SUMMARY+=("$1"); }

# =============================================================================
# 1) prostate_mri_anatomy  (MONAI bundle, Apache-2.0, ~306 MB)
# =============================================================================
setup_prostate_anatomy() {
  local sentinel="${ANATOMY_DIR}/models/model.ts"
  if [ "${FORCE}" -eq 0 ] && [ -e "${sentinel}" ]; then
    ok "prostate_mri_anatomy already present -> ${ANATOMY_DIR}"
    record "prostate_mri_anatomy: SKIPPED (already present)"
    return 0
  fi

  if [ "${FROM_LOCAL}" -eq 1 ] && [ -e "${LOCAL_ANATOMY}/models/model.ts" ]; then
    info "prostate_mri_anatomy: using local copy ${LOCAL_ANATOMY}"
    place_local "${LOCAL_ANATOMY}" "${ANATOMY_DIR}"
    record "prostate_mri_anatomy: from-local"
    return 0
  fi

  mkdir -p "${MODELS_DIR}"
  if have_cmd python && python -c 'import monai' >/dev/null 2>&1; then
    info "prostate_mri_anatomy: MONAI bundle download -> ${MODELS_DIR}"
    python -m monai.bundle download "prostate_mri_anatomy" --bundle_dir "${MODELS_DIR}"
  elif have_cmd huggingface-cli; then
    warn "monai not importable; falling back to huggingface-cli"
    huggingface-cli download MONAI/prostate_mri_anatomy --local-dir "${ANATOMY_DIR}"
  else
    err "Cannot fetch prostate_mri_anatomy: need python+monai or huggingface-cli."
    record "prostate_mri_anatomy: FAILED (no fetch tool)"
    return 1
  fi
  record "prostate_mri_anatomy: downloaded"
}

# =============================================================================
# 2) brats_mri_segmentation  (MONAI bundle, Apache-2.0, ~38 MB)
# =============================================================================
setup_brats() {
  local sentinel_ts="${BRATS_DIR}/models/model.ts"
  local sentinel_pt="${BRATS_DIR}/models/model.pt"
  if [ "${FORCE}" -eq 0 ] && { [ -e "${sentinel_ts}" ] || [ -e "${sentinel_pt}" ]; }; then
    ok "brats_mri_segmentation already present -> ${BRATS_DIR}"
    record "brats_mri_segmentation: SKIPPED (already present)"
    return 0
  fi

  mkdir -p "${MODELS_DIR}"
  if have_cmd python && python -c 'import monai' >/dev/null 2>&1; then
    info "brats_mri_segmentation: MONAI bundle download -> ${MODELS_DIR}"
    python -m monai.bundle download "brats_mri_segmentation" --bundle_dir "${MODELS_DIR}"
  elif have_cmd huggingface-cli; then
    warn "monai not importable; falling back to huggingface-cli"
    huggingface-cli download MONAI/brats_mri_segmentation --local-dir "${BRATS_DIR}"
  else
    err "Cannot fetch brats_mri_segmentation: need python+monai or huggingface-cli."
    record "brats_mri_segmentation: FAILED (no fetch tool)"
    return 1
  fi
  record "brats_mri_segmentation: downloaded"
}

# =============================================================================
# 3) prostate_mri_lesion_seg  (app + weights; NON-COMMERCIAL, ~895 MB weights)
#    App: Project-MONAI/research-contributions -> prostate-mri-lesion-seg
#    Weights: Google Drive folder (5 folds + classifier + organ = 7 files)
# =============================================================================
setup_lesion() {
  warn "prostate_mri_lesion_seg is NON-COMMERCIAL / no-clinical-use (NCI/NVIDIA license)."

  # ---- app code ----
  local app_ok=0
  if [ "${FORCE}" -eq 0 ] && [ -e "${LESION_APP_DIR}/app.py" ]; then
    ok "lesion app already present -> ${LESION_APP_DIR}"
    app_ok=1
  elif [ "${FROM_LOCAL}" -eq 1 ] && [ -e "${LOCAL_LESION_APP}/app.py" ]; then
    info "lesion app: using local copy ${LOCAL_LESION_APP}"
    place_local "${LOCAL_LESION_APP}" "${LESION_APP_DIR}"
    app_ok=1
  else
    if have_cmd git; then
      local tmp_clone; tmp_clone="$(mktemp -d)"
      info "lesion app: cloning ${RESEARCH_CONTRIB_REPO}"
      if git clone --depth 1 "${RESEARCH_CONTRIB_REPO}" "${tmp_clone}"; then
        local sub="${tmp_clone}/prostate-mri-lesion-seg/prostate_mri_lesion_seg_app"
        if [ -d "${sub}" ]; then
          place_local "${sub}" "${LESION_APP_DIR}"
          app_ok=1
        else
          err "app subdir not found in clone: ${sub}"
        fi
      else
        err "git clone of research-contributions failed."
      fi
      rm -rf "${tmp_clone}"
    else
      err "git not available; cannot fetch lesion app."
    fi
  fi

  # ---- weights (fold0..4 + classifier + organ) ----
  local weight_ok=0
  if [ "${FORCE}" -eq 0 ] && [ -e "${LESION_WEIGHT_DIR}/fold0/model_best_fold0.pth.tar" ]; then
    ok "lesion weights already present -> ${LESION_WEIGHT_DIR}"
    weight_ok=1
  elif [ "${FROM_LOCAL}" -eq 1 ] && [ -e "${LOCAL_LESION_WEIGHT}/fold0/model_best_fold0.pth.tar" ]; then
    info "lesion weights: using local copy ${LOCAL_LESION_WEIGHT}"
    place_local "${LOCAL_LESION_WEIGHT}" "${LESION_WEIGHT_DIR}"
    weight_ok=1
  else
    if have_cmd gdown; then
      info "lesion weights: gdown Google Drive folder (~895 MB) -> ${LESION_WEIGHT_DIR}"
      mkdir -p "${LESION_WEIGHT_DIR}"
      if gdown --folder "${LESION_GDRIVE_FOLDER}" -O "${LESION_WEIGHT_DIR}"; then
        weight_ok=1
      else
        err "gdown of lesion weights failed."
      fi
    else
      err "gdown not available; cannot fetch lesion weights."
      warn "Install with: pip install gdown ; then:"
      warn "  gdown --folder ${LESION_GDRIVE_FOLDER} -O ${LESION_WEIGHT_DIR}"
    fi
  fi

  if [ "${app_ok}" -eq 1 ] && [ "${weight_ok}" -eq 1 ]; then
    record "prostate_mri_lesion_seg: app + weights ready (NON-COMMERCIAL)"
    return 0
  fi
  # Report the partial state in the summary, but still fail: an empty or
  # half-populated lesion dir crashes detect_lesion_candidates much later with a
  # far less obvious error, so surface it here via a non-zero exit.
  record "prostate_mri_lesion_seg: PARTIAL (app=${app_ok} weights=${weight_ok})"
  return 1
}

# =============================================================================
# 4) cardiac_nnunet  (nnUNet RESULTS_FOLDER) -- SOFT / best-effort
#    Task900_ACDC_Phys : collaborator-trained (may be private; SURFdrive link
#                        inside ${CMR_REVERSE_ROOT}/download_nnunet_weights.sh)
#    Task027_ACDC       : public fallback (Isensee et al., Zenodo, CC BY-NC 4.0)
#    The cardiac-seg agent finalizes the exact backend/weights path; this step
#    never hard-fails the run.
# =============================================================================
setup_cardiac() {
  mkdir -p "${CARDIAC_RESULTS_DIR}"

  if [ "${FORCE}" -eq 0 ] && [ -n "$(ls -A "${CARDIAC_RESULTS_DIR}" 2>/dev/null || true)" ]; then
    ok "cardiac_nnunet results already populated -> ${CARDIAC_RESULTS_DIR}"
    record "cardiac_nnunet: SKIPPED (already present)"
    return 0
  fi

  case "${CARDIAC_SOURCE}" in
    task900)
      if [ -z "${CMR_REVERSE_ROOT}" ]; then
        warn "CARDIAC_SOURCE=task900 needs CMR_REVERSE_ROOT to point at the directory"
        warn "containing download_nnunet_weights.sh (collaborator-provided, may be private)."
        warn "For the public weights use CARDIAC_SOURCE=task027."
        record "cardiac_nnunet: UNRESOLVED (CMR_REVERSE_ROOT unset)"
        return 0
      fi
      local helper="${CMR_REVERSE_ROOT}/download_nnunet_weights.sh"
      if [ -f "${helper}" ]; then
        info "cardiac: running collaborator helper ${helper}"
        info "  (sets up Task900_ACDC_Phys under RESULTS_FOLDER/nnUNet/2d)"
        if RESULTS_FOLDER="${CARDIAC_RESULTS_DIR}" bash "${helper}"; then
          record "cardiac_nnunet: Task900_ACDC_Phys (via cmr_reverse helper)"
        else
          warn "Task900 helper failed (link may be private). Try CARDIAC_SOURCE=task027."
          record "cardiac_nnunet: UNRESOLVED (Task900 helper failed)"
        fi
      else
        warn "cmr_reverse helper not found: ${helper}"
        warn "Task900_ACDC_Phys may be private; use CARDIAC_SOURCE=task027 for the public fallback."
        record "cardiac_nnunet: UNRESOLVED (no Task900 helper)"
      fi
      ;;
    task027)
      if have_cmd curl; then
        local zip="${CARDIAC_DIR}/Task027_ACDC.zip"
        info "cardiac: downloading public Task027_ACDC (~1.8 GB) from Zenodo"
        if curl -L --fail --show-error -o "${zip}" "${TASK027_URL}"; then
          if have_cmd unzip; then
            # nnUNet v1 expects RESULTS_FOLDER/nnUNet/<config>/TaskXXX/<trainer>.
            # The Zenodo Task027_ACDC archive expands to <config>/TaskXXX/... with
            # no leading nnUNet/ component, so extract into the nnUNet/ subdir.
            mkdir -p "${CARDIAC_RESULTS_DIR}/nnUNet"
            info "cardiac: unzipping Task027_ACDC -> ${CARDIAC_RESULTS_DIR}/nnUNet"
            unzip -q "${zip}" -d "${CARDIAC_RESULTS_DIR}/nnUNet" || warn "unzip reported issues; inspect ${zip}"
            rm -f "${zip}"
            record "cardiac_nnunet: Task027_ACDC (public CC BY-NC 4.0)"
          else
            warn "unzip not available; left archive at ${zip}"
            record "cardiac_nnunet: Task027 downloaded (unzip manually)"
          fi
        else
          warn "Task027 download failed."
          record "cardiac_nnunet: UNRESOLVED (Task027 download failed)"
        fi
      else
        err "curl not available; cannot fetch Task027_ACDC."
        record "cardiac_nnunet: UNRESOLVED (no curl)"
      fi
      ;;
    auto|*)
      warn "cardiac_nnunet: no source auto-selected (CARDIAC_SOURCE=auto)."
      cat <<EOF
  The cardiac backend/weights are finalized by the separate cardiac-seg agent.
  Two options -- pick one and re-run with CARDIAC_SOURCE set:

  (A) Collaborator model Task900_ACDC_Phys (MICCAI 2025 'Reverse Imaging'):
        CARDIAC_SOURCE=task900 $0 --only cardiac
      Uses ${CMR_REVERSE_ROOT}/download_nnunet_weights.sh (SURFdrive link;
      may be PRIVATE -- requires collaborator access).

  (B) Public fallback Task027_ACDC (Isensee et al., Zenodo, CC BY-NC 4.0):
        CARDIAC_SOURCE=task027 $0 --only cardiac
      curl -L -o Task027_ACDC.zip "${TASK027_URL}"

  RESULTS_FOLDER target: ${CARDIAC_RESULTS_DIR}
EOF
      record "cardiac_nnunet: DEFERRED (choose CARDIAC_SOURCE=task900|task027)"
      ;;
  esac
}

# =============================================================================
# checkpoints/ note -- prostate_distortion was DROPPED
# =============================================================================
setup_checkpoints_note() {
  mkdir -p "${CHECKPOINTS_DIR}"
  local note="${CHECKPOINTS_DIR}/README_REMOVED.txt"
  if [ ! -e "${note}" ]; then
    cat > "${note}" <<'EOF'
prostate_distortion checkpoints have been REMOVED.

That tool (prostate distortion / diffusion recovery) was dropped from the
public release. No checkpoints are expected under assets/checkpoints/.

If a future tool needs checkpoints, document its source + license in
docs/ASSETS.md and add a fetch branch to scripts/download_assets.sh.
EOF
    info "wrote ${note}"
  fi
}

# =============================================================================
# main
# =============================================================================
main() {
  info "Repo root : ${REPO_ROOT}"
  info "Assets    : ${ASSETS_DIR}"
  [ "${FROM_LOCAL}" -eq 1 ] && info "Mode      : from-local (fallback to network)"
  [ -n "${ONLY}" ] && info "Selection : ${ONLY}"

  mkdir -p "${MODELS_DIR}" "${CHECKPOINTS_DIR}"

  local rc=0
  want prostate && { setup_prostate_anatomy || rc=1; }
  want brats    && { setup_brats            || rc=1; }
  want lesion   && { setup_lesion           || rc=1; }
  want cardiac  && { setup_cardiac          || true; }   # cardiac never fails run

  setup_checkpoints_note

  echo
  echo "==================== SUMMARY ===================="
  # NOTE: "${SUMMARY[@]}" on an empty array is an unbound-variable error under
  # `set -u` on bash < 4.4 (macOS ships 3.2), so guard both expansions.
  if [ "${#SUMMARY[@]:-0}" -eq 0 ]; then
    echo "  (nothing selected)"
  else
    for line in "${SUMMARY[@]+"${SUMMARY[@]}"}"; do
      echo "  - ${line}"
    done
  fi
  echo "-------------------------------------------------"
  if [ -d "${ASSETS_DIR}" ]; then
    echo "  Total assets/ size:"
    du -sh "${ASSETS_DIR}" 2>/dev/null | sed 's/^/    /' || true
    for d in "${ANATOMY_DIR}" "${BRATS_DIR}" "${LESION_DIR}" "${CARDIAC_DIR}"; do
      [ -e "${d}" ] && du -sh "${d}" 2>/dev/null | sed 's/^/    /' || true
    done
  fi
  echo "================================================="
  echo
  echo "Point the tools at assets/ via env vars (see docs/ASSETS.md):"
  echo "  export MRI_AGENT_MODEL_REGISTRY=${MODELS_DIR}"
  echo "  export MRI_AGENT_LESION_APP_DIR=${LESION_APP_DIR}"
  echo "  export MRI_AGENT_LESION_WEIGHTS_DIR=${LESION_WEIGHT_DIR}"
  echo "  export MRI_AGENT_CARDIAC_RESULTS_FOLDER=${CARDIAC_RESULTS_DIR}"

  return "${rc}"
}

main "$@"
