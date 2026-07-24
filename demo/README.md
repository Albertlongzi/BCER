# BCER Demo — Longest Tool Chain per Domain (open data)

This demo showcases the **longest BCER tool chain** for each of the three
supported domains — **brain**, **cardiac**, **prostate** — using **public,
open** MRI datasets.

**No medical data is committed to this repository.** The repo `.gitignore`
excludes `*.nii`, `*.nii.gz`, `*.dcm`, `*.h5` (and derived arrays), so the
three `demo/cases/*/` directories ship as empty placeholders (each holds a
`.gitkeep`). You populate them **at run time** from a local data root with
`build_demo_cases.sh`. The staged files stay untracked by git.

```
demo/
├── build_demo_cases.sh      # stage the 3 cases from your local data root
├── run_longest_chain.sh     # build manifest + run the longest chain per domain
├── cases/
│   ├── Brats18_CBICA_AAM_1/          (.gitkeep) — brain  (BraTS 2018)
│   ├── sub-019_2/                    (.gitkeep) — prostate (fastMRI Prostate)
│   └── acdc_multiseq_patient061_ed/  (.gitkeep) — cardiac (ACDC)
└── README.md
```

---

## The three demo cases

| Domain | Case dir | Dataset | Input | Longest chain demonstrated |
| --- | --- | --- | --- | --- |
| brain | `cases/Brats18_CBICA_AAM_1/` | BraTS 2018 | 4-modality NIfTI (T1, T1c, T2, FLAIR) | `identify → register(T2,FLAIR→T1c) → segment → features → classify-grade → package → report` |
| prostate | `cases/sub-019_2/` | fastMRI Prostate | DICOM series (T2w, ADC, DWI-trace, calc-b) | `identify → register(ADC,DWI→T2w) → segment → features(×2) → detect-lesions → package → report` |
| cardiac | `cases/acdc_multiseq_patient061_ed/` | ACDC | cine NIfTI (single ED frame) | `identify → segment-cine → classify-disease → features → qa → package → report` |

The chains above are the full-pipeline templates in
`agent/plans/templates/{brain,prostate,cardiac}_full_pipeline.json`.

---

## Data provenance, licensing & registration

None of these datasets are redistributed here — obtain each yourself.

### Brain — BraTS 2018 (`Brats18_CBICA_AAM_1`)
- **Source:** MICCAI Brain Tumor Segmentation (BraTS) 2018 Challenge, Validation
  set. Multi-institutional pre-operative glioma MRI; four co-registered,
  skull-stripped modalities per case (T1, T1c/`t1ce`, T2, FLAIR), 1 mm³.
- **Access / registration:** BraTS data is distributed via the challenge portal
  (historically CBICA IPP / Synapse; recent years via
  <https://www.synapse.org/brats>). Registration and agreement to the challenge
  data-use terms are required. Cite Menze et al. 2015, Bakas et al. 2017.
- **Note:** this Validation case has **no** ground-truth segmentation. A Training
  case (e.g. `HGG/Brats18_2013_10_1`) additionally ships `*_seg.nii` if you want
  segmentation ground truth.

### Prostate — fastMRI Prostate (`sub-019_2`)
- **Source:** fastMRI Prostate (NYU Langone / Meta AI FAIR). Bi-parametric
  prostate MRI: axial T2w plus diffusion (trace/high-b, calculated b-value, ADC).
- **Access / registration:** requires the fastMRI data-use agreement at
  <https://fastmri.med.nyu.edu/>. This single example is included in the demo
  with the data owner's explicit confirmation that it is publishable.
- **Layout staged:** DICOM series subfolders (`AX_T2`, `AX_DIFFUSION_ADC`,
  `AX_DIFFUSION_TRACEW`, `AX_DIFFUSION_CALC_BVAL`).

### Cardiac — ACDC (`acdc_multiseq_patient061_ed`)
- **Source:** Automated Cardiac Diagnosis Challenge (ACDC), MICCAI 2017, CREATIS,
  University of Lyon. Short-axis cine MRI with disease-group labels.
- **Access / download:** <https://www.creatis.insa-lyon.fr/Challenge/acdc>
  (free registration). Cite Bernard et al. 2018.
- **Layout staged:** the **single end-diastole (ED) frame** as the cine NIfTI
  (`patient061_frame01_0000.nii.gz`) plus its ground-truth segmentation
  (`patient061_frame01_gt.nii.gz`, excluded from sequence identification).

---

## Cardiac 4D-cine gap (be honest)

The `long_cardiac_full` contract and the segmentation tests expect a **4D cine**
volume (e.g. `patient061_4d.nii.gz`). **That 4D volume is not on local disk.**
What is available locally is the **single ED frame** (a 3D volume), plus the ES
frame (`patient061_frame10_0000.nii.gz`) and ED ground truth. The demo stages the
**ED frame only** as the CINE input so sequence resolution is unambiguous, and
the cardiac chain runs on that single 3D frame.

`identify_sequences` maps `patient061_frame01*` to the `CINE` sequence, so the
identify → segment → classify → report chain proceeds. If a downstream tool
strictly requires 4D input, download the full ACDC cine from
<https://www.creatis.insa-lyon.fr/Challenge/acdc>, assemble
`patient061_4d.nii.gz`, and stage it into
`demo/cases/acdc_multiseq_patient061_ed/` (set `CARDIAC_SRC` to point at it).

---

## How to run

### 1. Populate the demo cases

There are no built-in dataset paths — point the script at your own copies.
`DATA_ROOT` supplies the brain + prostate sources; the ACDC tree is named
separately via `CARDIAC_SRC`. Any domain you do not have can be skipped.

```bash
# brain + prostate from one root, cardiac from its own ACDC tree:
DATA_ROOT=/path/to/open_data \
CARDIAC_SRC=/path/to/acdc/TaskDir \
  demo/build_demo_cases.sh

# Or name every source explicitly:
BRAIN_SRC=/path/to/BraTS/CaseX \
PROSTATE_SRC=/path/to/prostate/subZ/DICOMS \
CARDIAC_SRC=/path/to/acdc/TaskDir \
  demo/build_demo_cases.sh

# Stage only the domains you have:
DATA_ROOT=/path/to/open_data demo/build_demo_cases.sh --skip-cardiac
```

The script exits with an error listing exactly which domain is missing a
source, rather than silently probing paths you never named.

The script is idempotent (each case dir is cleared except its `.gitkeep` and
re-staged), prints exactly what it stages, and converts BraTS `.nii` → `.nii.gz`
to match the canonical layout (the tools accept both). gzip is CPU-only; no GPU
is needed for staging.

### 2. Build the manifest and run the longest chain

```bash
# Build the benchmark manifest from demo/cases only (safe; no runner needed):
demo/run_longest_chain.sh --build-manifest-only

# Print the intended per-domain runner commands (dry run, default):
demo/run_longest_chain.sh

# Actually execute (requires a live OpenAI-compatible inference server —
# see benchmark/README.md):
demo/run_longest_chain.sh --execute
```

`run_longest_chain.sh` runs `scripts/manifest_builder.py` over the three demo
case dirs to produce `demo/cases_manifest.jsonl` (git-ignored), then invokes
`benchmark/benchmark_runner.py --arm bcer` on the longest task per domain.

> **Prerequisite:** `--execute` needs an OpenAI-compatible chat-completions
> endpoint reachable at `--server-base-url` (any vLLM / llama.cpp / SGLang
> server will do; the tools do the imaging work, the model only plans and
> writes the report). Manifest building works independently and needs no
> server.

### Longest task per domain

| Domain | Task id (registry) | Chain length |
| --- | --- | --- |
| prostate | `long_prostate_full` | 5–7 steps (identify → register → segment → lesion → features → report) |
| cardiac | `long_cardiac_full` | 3–4 steps (identify → segment → classify → report) |
| brain | `long_brain_full` | 6 steps (identify → segment → features → grade-classify → package → report) |

Override any of these with `PROSTATE_TASK=<id>` / `CARDIAC_TASK=<id>` /
`BRAIN_TASK=<id>`. The longest brain *template* is the 8-node
`agent/plans/templates/brain_full_pipeline.json`; `long_brain_full` is the
runner-backed contract that scores the same pipeline. BraTS inputs arrive
already co-registered at 1×1×1 mm, so the planner normally skips the
registration stage on that data.
