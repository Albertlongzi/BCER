# Datasets

BCER does not ship any medical imaging data. To run the benchmark on your own
machine you need to:

1. Obtain one or more public MRI datasets.
2. Lay the cases out in a directory the manifest builder can scan.
3. Run `scripts/manifest_builder.py` to produce `cases_manifest.jsonl`.
4. Pass that manifest to `benchmark/benchmark_runner.py`.

This document covers all four steps.

---

## 1. Supported domains and dataset shape

The benchmark currently supports three domains. Each task contract in
`configs/tasks_registry.json` declares its domain and required modalities; the
manifest builder marks each case with the tasks it can serve.

| Domain | Reference modality | Other modalities | Public dataset examples |
| --- | --- | --- | --- |
| `prostate` | T2w | ADC, DWI (high-b), T1c (optional) | ProstateX, Prostate-MRI-US-Biopsy, Picai |
| `brain` | T1c (or T1) | T1, T2, FLAIR | BraTS (any year) |
| `cardiac` | cine SAX | (none, single sequence) | ACDC, M&Ms; raw k-space H5 supported separately |

The scanner accepts NIfTI (`.nii`, `.nii.gz`), DICOM (`.dcm`), and cardiac raw
k-space H5 (`.h5`, `.hdf5`) inputs. Modality detection looks at file names and
DICOM headers; it does not require a specific naming convention but matches
the obvious ones (`T2w`, `ADC`, `DWI`, `T1c`, `t1ce`, `flair`, etc.).

---

## 2. Directory layout per domain

The manifest builder scans **case roots** — directories where each immediate
subdirectory is one case. Layout examples:

### Prostate (DICOM or NIfTI)

```
$BCER_PROSTATE_ROOT/
├── sub-001/
│   ├── T2w.nii.gz
│   ├── ADC.nii.gz
│   └── DWI_b1500.nii.gz
├── sub-002/
│   └── DICOM/  (any DICOM series tree the identify_sequences tool can parse)
└── ...
```

### Brain (NIfTI; BraTS layout)

```
$BCER_BRATS_ROOT/
├── Brats18_CBICA_AAM_1/
│   ├── Brats18_CBICA_AAM_1_t1.nii.gz
│   ├── Brats18_CBICA_AAM_1_t1ce.nii.gz
│   ├── Brats18_CBICA_AAM_1_t2.nii.gz
│   └── Brats18_CBICA_AAM_1_flair.nii.gz
└── ...
```

### Cardiac cine (NIfTI; ACDC layout)

```
$BCER_ACDC_ROOT/
├── patient001_ed/
│   └── patient001_ed.nii.gz
├── patient001_es/
│   └── patient001_es.nii.gz
└── ...
```

### Cardiac raw k-space (HDF5)

Each `.h5` or `.hdf5` file is treated as one case. Pass the **directory** that
contains the files; the builder enumerates them.

```
$BCER_CARDIAC_RAW_ROOT/
├── caseA.h5
├── caseB.h5
└── ...
```

Set the environment variables that the README and the benchmark examples
reference:

```bash
export BCER_PROSTATE_ROOT=/path/to/prostate
export BCER_BRATS_ROOT=/path/to/brats
export BCER_ACDC_ROOT=/path/to/acdc
export BCER_CARDIAC_RAW_ROOT=/path/to/cardiac_raw   # optional
```

---

## 3. Build the manifest

```bash
python scripts/manifest_builder.py \
  --prostate-root "$BCER_PROSTATE_ROOT" \
  --brain-root    "$BCER_BRATS_ROOT" \
  --cardiac-root  "$BCER_ACDC_ROOT" \
  --output        benchmark/cases_manifest.jsonl
```

Useful caps when developing or sanity-checking:

| Flag | Effect |
| --- | --- |
| `--max-cases N` | Stop after N total cases across all domains |
| `--max-prostate-cases N` | Per-domain cap |
| `--max-brain-cases N` | Per-domain cap |
| `--max-cardiac-cases N` | Per-domain cap |
| `--max-depth K` | Limit recursive scan depth per case root (default: scan deeply) |
| `--max-files-per-case N` | Cap files scanned per case (large DICOM trees) |

Each line in the output manifest is one JSON object:

```json
{
  "case_id": "prostate__sub-001",
  "domain": "prostate",
  "case_root": "/abs/path/to/sub-001",
  "input_format": "nifti",
  "modalities": {"t2w": true, "adc": true, "dwi": true, "t1c": false},
  "supports_tasks": ["short_denoise", "short_superres", "medium_register_prostate",
                     "long_prostate_full"]
}
```

`supports_tasks` is computed by matching the case's detected modalities
against each task contract's `required_modalities`. The benchmark runner uses
this to skip cases that cannot serve a requested task.

---

## 4. Use the manifest

Run one `(task, arm, fault)` cell:

```bash
python benchmark/benchmark_runner.py \
  --manifest benchmark/cases_manifest.jsonl \
  --task    long_prostate_full \
  --arm     bcer \
  --fault   none \
  --runs-root runs
```

The runner reads the manifest, filters to cases supporting the requested task,
and dispatches one run per case. Results land under `runs/` (one directory per
case+task+arm+fault combination) and a per-cell summary under `benchmark/`.

See `benchmark/README.md` for full CLI usage and
`docs/METRICS.md` for what the output numbers mean.

---

## 5. Task IDs

| Task | Domain | Pipeline length | Description |
| --- | --- | --- | --- |
| `short_denoise` | prostate / brain | 1 step | BM3D denoise a single modality |
| `short_superres` | prostate / brain | 1 step | Resample to a finer grid |
| `short_segment_brain` | brain | 1 step | BraTS tumour subregion segmentation |
| `short_recon_grappa` | cardiac (raw k-space) | 1 step | GRAPPA reconstruction or H5 → NIfTI |
| `medium_register_prostate` | prostate | 2 steps | Identify sequences → register ADC/DWI → T2w |
| `medium_brain_grade_classify` | brain | 2 steps | Segment tumour → grade HGG/LGG |
| `long_prostate_full` | prostate | 5–7 steps | Identify → register → segment → features → optional lesion → report |
| `long_cardiac_full` | cardiac | 3–4 steps | (Optional reconstruction) → segment → classify → report |

Modality requirements per task live under each contract's
`required_modalities` field in `configs/tasks_registry.json`.

---

## 6. Bring your own dataset

If your data does not match the layouts above, you have two options.

**Option A — write a tiny wrapper.** Copy your data into the expected layout
or symlink it. The manifest builder is path-agnostic; it only reads file
extensions and DICOM headers.

**Option B — emit the manifest yourself.** Each manifest line is just a JSON
object with the five fields shown in §3. If you can compute those fields from
your own metadata, you can write `cases_manifest.jsonl` directly without
running `manifest_builder.py`. The benchmark runner does not care how the
manifest was produced.

In both cases, make sure `domain` matches one of `prostate`, `brain`,
`cardiac`, and that `modalities` keys match what the task contracts in
`configs/tasks_registry.json` expect.
