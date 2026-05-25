#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MRI_AGENT_MODEL_REGISTRY:-${REPO_ROOT}/models}"
TARGET="${1:-all}"
PROSTATE_VERSION="0.3.6"
BRATS_VERSION="0.5.2"

mkdir -p "${MODEL_DIR}"

download_bundle() {
  local name="$1"
  local version="$2"
  echo "[INFO] Downloading ${name} (version ${version}) into ${MODEL_DIR}"
  python -m monai.bundle download "${name}" --version "${version}" --bundle_dir "${MODEL_DIR}"
}

case "${TARGET}" in
  all)
    download_bundle "prostate_mri_anatomy" "${PROSTATE_VERSION}"
    download_bundle "brats_mri_segmentation" "${BRATS_VERSION}"
    ;;
  prostate)
    download_bundle "prostate_mri_anatomy" "${PROSTATE_VERSION}"
    ;;
  brats)
    download_bundle "brats_mri_segmentation" "${BRATS_VERSION}"
    ;;
  *)
    echo "Usage: $0 [all|prostate|brats]"
    exit 2
    ;;
 esac

echo "[OK] Bundles ready under: ${MODEL_DIR}"
