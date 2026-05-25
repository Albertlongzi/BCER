#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXTERNAL_ROOT="${MRI_AGENT_LESION_ROOT:-${REPO_ROOT}/external/prostate_mri_lesion_seg}"
RESEARCH_ROOT="${EXTERNAL_ROOT}/scripts/research-contributions"
WEIGHTS_DIR="${MRI_AGENT_LESION_WEIGHTS_DIR:-${EXTERNAL_ROOT}/weight}"

mkdir -p "${EXTERNAL_ROOT}/scripts"

if [ -d "${RESEARCH_ROOT}/.git" ]; then
  echo "[INFO] MONAI research-contributions already present: ${RESEARCH_ROOT}"
else
  echo "[INFO] Cloning MONAI research-contributions into ${RESEARCH_ROOT}"
  git clone https://github.com/Project-MONAI/research-contributions.git "${RESEARCH_ROOT}"
fi

mkdir -p "${WEIGHTS_DIR}"

cat <<'TXT'
[OK] Repository ready.

Next step: download the prostate-mri-lesion-seg RRUNet3D weights and place them here:
  $MRI_AGENT_LESION_WEIGHTS_DIR (or default: <repo>/external/prostate_mri_lesion_seg/weight)

Expected files:
  fold0/model_best_fold0.pth.tar
  fold1/model_best_fold1.pth.tar
  fold2/model_best_fold2.pth.tar
  fold3/model_best_fold3.pth.tar
  fold4/model_best_fold4.pth.tar

If you keep the repo elsewhere, set:
  MRI_AGENT_LESION_APP_DIR to .../prostate_mri_lesion_seg_app
  MRI_AGENT_LESION_WEIGHTS_DIR to .../weight
TXT
