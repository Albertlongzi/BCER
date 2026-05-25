# Demo Case Stubs

These directories are intentionally empty placeholders so the shell's
domain-switching auto-load logic resolves a default case name when no real
data is present.

To use the demo flow with real data, replace each placeholder with a case
directory that follows the expected layout for that domain:

| Domain | Stub | Expected contents |
| --- | --- | --- |
| prostate | `cases/sub-019_2/` | DICOM series or NIfTI volumes (T2w, ADC, DWI) |
| brain | `cases/Brats18_CBICA_AAM_1/` | BraTS-style 4-modality NIfTI (T1, T1c, T2, FLAIR) |
| cardiac | `cases/acdc_multiseq_patient061_ed/` | ACDC-style cine NIfTI |
