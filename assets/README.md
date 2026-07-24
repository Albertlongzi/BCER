# BCER Local Assets

All tool code, external tool adapters, and checkpoints for the public release
should live under this project-local `assets/` directory.

Expected layout:

```text
assets/
  models/
    prostate_mri_anatomy/
      models/model.ts
    brats_mri_segmentation/
    prostate_mri_lesion_seg/
      weight/fold0/model_best_fold0.pth.tar
      weight/fold1/model_best_fold1.pth.tar
      weight/fold2/model_best_fold2.pth.tar
      weight/fold3/model_best_fold3.pth.tar
      weight/fold4/model_best_fold4.pth.tar
    cardiac_nnunet/
      results/
  checkpoints/
  external/
    cmr_reverse/
```

Large checkpoints are intentionally not tracked in git. The release should
document download links and licenses before enabling full inference workflows.
