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
    prostate_distortion/
      diff_t2cnn_clean_epoch_092.pt
      mageultra_epoch_025.pt
  external/
    cmr_reverse/
    Prostate_distortion_recover/
```

Large checkpoints are intentionally not tracked in git. The release should
document download links and licenses before enabling full inference workflows.
