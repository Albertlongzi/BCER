# Model Assets

BCER_open ships **without** model weights. This document lists every model the
tools need, where it comes from, how to fetch it, its approximate size, its
license, and the exact target path under `assets/`.

For a one-command setup use:

```bash
# from the repo root
scripts/download_assets.sh                 # download everything from the network
scripts/download_assets.sh --from-local    # prefer known local HPC copies, download the rest
scripts/download_assets.sh --only prostate,brats
scripts/download_assets.sh --help
```

The script is idempotent (already-present models are skipped unless `--force`),
prints a size summary, and never hard-fails the whole run because of the
cardiac model (see the cardiac notes below).

---

## Target layout

```text
assets/
  models/
    prostate_mri_anatomy/               # MONAI bundle; tool reads models/model.ts
      models/model.ts
    brats_mri_segmentation/             # MONAI bundle
      models/model.ts
    prostate_mri_lesion_seg/            # NON-COMMERCIAL
      prostate_mri_lesion_seg_app/      # app code (rrunet3D.py, app.py, ...)
      weight/
        fold0/model_best_fold0.pth.tar
        fold1/model_best_fold1.pth.tar
        fold2/model_best_fold2.pth.tar
        fold3/model_best_fold3.pth.tar
        fold4/model_best_fold4.pth.tar
        classifier/model_best.pth.tar
        organ/model.ts
    cardiac_nnunet/                     # nnUNet RESULTS_FOLDER (see cardiac note)
      results/
  checkpoints/                          # prostate_distortion REMOVED (tool dropped)
```

---

## Model table

| Model | Tool(s) that need it | Source URL | Download command | Approx size | License | Target path |
|---|---|---|---|---|---|---|
| **prostate_mri_anatomy** | `prostate_segmentation` | https://huggingface.co/MONAI/prostate_mri_anatomy | `python -m monai.bundle download "prostate_mri_anatomy" --bundle_dir assets/models`<br>fallback: `huggingface-cli download MONAI/prostate_mri_anatomy --local-dir assets/models/prostate_mri_anatomy` | ~306 MB | Apache-2.0 | `assets/models/prostate_mri_anatomy/` |
| **brats_mri_segmentation** | `brats_mri_segmentation` (brain tumor seg) | https://huggingface.co/MONAI/brats_mri_segmentation | `python -m monai.bundle download "brats_mri_segmentation" --bundle_dir assets/models`<br>fallback: `huggingface-cli download MONAI/brats_mri_segmentation --local-dir assets/models/brats_mri_segmentation` | ~38 MB | Apache-2.0 | `assets/models/brats_mri_segmentation/` |
| **prostate_mri_lesion_seg** (app) | `detect_lesion_candidates` | https://github.com/Project-MONAI/research-contributions (subdir `prostate-mri-lesion-seg`) | `git clone --depth 1 https://github.com/Project-MONAI/research-contributions.git` then copy `prostate-mri-lesion-seg/prostate_mri_lesion_seg_app/` | small | **NON-COMMERCIAL** (NCI/NVIDIA) | `assets/models/prostate_mri_lesion_seg/prostate_mri_lesion_seg_app/` |
| **prostate_mri_lesion_seg** (weights) | `detect_lesion_candidates` | https://drive.google.com/drive/folders/1EpjrlzEdV7CcaCYqGTIEzOapamP4Ag6M | `gdown --folder https://drive.google.com/drive/folders/1EpjrlzEdV7CcaCYqGTIEzOapamP4Ag6M -O assets/models/prostate_mri_lesion_seg/weight` | ~895 MB (7 files) | **NON-COMMERCIAL** (NCI/NVIDIA) | `assets/models/prostate_mri_lesion_seg/weight/` |
| **cardiac Task900_ACDC_Phys** (collaborator) | `cardiac_cine_segmentation` | SURFdrive link inside `/common/longz2/cmr_reverse/download_nnunet_weights.sh` | `RESULTS_FOLDER=assets/models/cardiac_nnunet/results bash /common/longz2/cmr_reverse/download_nnunet_weights.sh` | ~n/a | Research-use (collaborator; **may be PRIVATE**) | `assets/models/cardiac_nnunet/results/nnUNet/2d/` |
| **cardiac Task027_ACDC** (public fallback) | `cardiac_cine_segmentation` | https://zenodo.org/records/3734294 (DOI 10.5281/zenodo.3734294, `Task027_ACDC.zip`) | `curl -L -o Task027_ACDC.zip "https://zenodo.org/records/3734294/files/Task027_ACDC.zip?download=1"` then unzip into `assets/models/cardiac_nnunet/results/` | ~1.8 GB | **CC BY-NC 4.0** | `assets/models/cardiac_nnunet/results/` |

---

## Cardiac model note (unresolved)

The framework's cardiac backend is **not finalized here** — the separate
cardiac-seg agent chooses the exact backend/weights. Two options exist:

- **Task900_ACDC_Phys** — collaborator-trained model (MICCAI 2025 *Reverse
  Imaging*, TU Delft / Tao lab). Fetched by the helper
  `/common/longz2/cmr_reverse/download_nnunet_weights.sh` (a SURFdrive link).
  **This may be private / require collaborator access.**
- **Task027_ACDC** — the reliable **public** fallback (Isensee et al., nnU-Net
  ACDC baseline), on Zenodo under CC BY-NC 4.0.

Select one at fetch time:

```bash
CARDIAC_SOURCE=task900 scripts/download_assets.sh --only cardiac   # collaborator (may be private)
CARDIAC_SOURCE=task027 scripts/download_assets.sh --only cardiac   # public fallback (~1.8 GB)
```

With `CARDIAC_SOURCE=auto` (default) the script only prints these options and
does not download; it never fails the overall run when cardiac is unresolved.

---

## Pointing the tools at `assets/`

The tools resolve models from `assets/models/` by default in several cases, but
you can make it explicit (and override the legacy `./models` and
`./external/...` defaults) with environment variables:

```bash
export MRI_AGENT_MODEL_REGISTRY="$PWD/assets/models"                 # prostate_mri_anatomy + brats
export MRI_AGENT_LESION_APP_DIR="$PWD/assets/models/prostate_mri_lesion_seg/prostate_mri_lesion_seg_app"
export MRI_AGENT_LESION_WEIGHTS_DIR="$PWD/assets/models/prostate_mri_lesion_seg/weight"
export MRI_AGENT_CARDIAC_RESULTS_FOLDER="$PWD/assets/models/cardiac_nnunet/results"
```

Individual overrides also exist: `MRI_AGENT_PROSTATE_BUNDLE_DIR`,
`MRI_AGENT_BRATS_BUNDLE_DIR`.

### Cardiac backend variables

`segment_cardiac_cine` shells out to the vendored nnUNet fork, so it has its own
set of variables. All are optional; the defaults below are what the tool uses
when the variable is unset.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MRI_AGENT_CARDIAC_BACKEND_ROOT` | `./external/nnunet_phys_seg` | vendored nnUNet fork (code only) |
| `MRI_AGENT_CARDIAC_SEG_PYTHON` | current interpreter | python that has the nnUNet deps installed (see `envs/cardiac.yml`) |
| `MRI_AGENT_CARDIAC_RESULTS_FOLDER` | `assets/models/cardiac_nnunet/results` | nnUNet `RESULTS_FOLDER` |
| `MRI_AGENT_CARDIAC_TASK` | `Task900_ACDC_Phys` | nnUNet task name |
| `MRI_AGENT_CARDIAC_TRAINER` | `nnUNetTrainerV2_InvGreAug` | nnUNet trainer class |

The `MRI_AGENT_CARDIAC_CMR_REVERSE_ROOT` variable and the `cmr_reverse_root`
tool argument remain accepted as aliases of `..._BACKEND_ROOT` / `backend_root`.

**If you installed the public Task027_ACDC fallback** rather than the
collaborator-trained Task900 weights, you must also switch the task and trainer
— otherwise the tool asks nnUNet for a model that is not on disk:

```bash
export MRI_AGENT_CARDIAC_TASK=Task027_ACDC
export MRI_AGENT_CARDIAC_TRAINER=nnUNetTrainerV2
```

---

## Removed assets

`assets/checkpoints/` intentionally holds **no** weights. The
`prostate_distortion` tool (diffusion-based distortion recovery) was dropped
from the public release, so its checkpoints
(`diff_t2cnn_clean_epoch_092.pt`, `mageultra_epoch_025.pt`) are **not** fetched
and **not** expected.

---

## ⚠️ LICENSE NOTES — read before use

1. **prostate_mri_lesion_seg is NON-COMMERCIAL and NOT for clinical use.**
   The weights and app are covered by a custom NCI/NVIDIA research license
   (*Prostate-MRI_Lesion_Detection*), **not** Apache-2.0. Per that license:
   **"THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS OF HUMAN
   SUBJECTS."** Redistribution is allowed only under the same terms. Do not use
   it commercially or for patient care.

2. **Cardiac ACDC weights (Task027_ACDC) are CC BY-NC 4.0** — non-commercial,
   attribution required (Isensee et al.; Zenodo DOI 10.5281/zenodo.3734294).
   The collaborator model **Task900_ACDC_Phys** is research-use and may be
   access-restricted.

3. **Research use only, overall.** Even the Apache-2.0 MONAI bundles
   (`prostate_mri_anatomy`, `brats_mri_segmentation`) are research/education
   tools and are **not** cleared medical devices. Nothing in this repository is
   validated for clinical decision-making. The most restrictive license among
   the assets you install governs your overall use.
