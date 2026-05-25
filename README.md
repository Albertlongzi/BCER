<h1 align="center">BCER</h1>

<p align="center">
  <b>Bounded Cerebellum Execution Runtime</b><br>
  An agent framework for reliable execution of long-horizon MRI analysis workflows.
</p>

<p align="center">
  <a href="#quickstart"><img alt="Quickstart" src="https://img.shields.io/badge/quickstart-5%20min-2ea44f"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-paper%20release-orange">
</p>

---

BCER separates **planning** (constrained sketch over a tool catalogue) from
**execution** (a deterministic cerebellum runtime that binds symbolic artifacts
to concrete files) and adds **bounded local recovery** (a two-tier reflector
that repairs recoverable failures and halts on nonrecoverable ones).

This repository accompanies the MICCAI paper. It contains:

- the BCER agent controller, planner, compiler, and reflector,
- a strict tool registry with **21 MRI-domain tools** across prostate, brain, and cardiac workflows,
- a **benchmark harness** with fault injection (8 fault types) and 4 controller arms,
- a **smoke benchmark** that proves install correctness without medical data.

```
                   ┌──────────────────────────────┐
   user goal ────▶ │  Planner  (constrained sketch) │
                   └──────────────┬───────────────┘
                                  │  sketch JSON
                                  ▼
                   ┌──────────────────────────────┐
                   │  Compiler (sketch → DAG)      │
                   └──────────────┬───────────────┘
                                  │  validated DAG
                                  ▼
                   ┌──────────────────────────────┐         ┌──────────────────┐
                   │  Cerebellum (executor)        │ ──────▶ │  Tool registry    │
                   └──────────────┬───────────────┘         └──────────────────┘
                                  │  on failure
                                  ▼
                   ┌──────────────────────────────┐
                   │  Reflector  Tier-1 (rules)    │
                   │             Tier-2 (LLM)      │
                   └──────────────────────────────┘
```

---

## Quickstart

Five minutes, no medical data, no GPU, no model weights.

```bash
git clone https://github.com/Albertlongzi/BCER.git
cd BCER
conda env create -f envs/base.yml
conda activate bcer-base
pip install -e .

python -m benchmark.smoke
```

Expected last lines:

```
[smoke] step 1/3 dummy_load_case        OK
[smoke] step 2/3 dummy_segment          OK
[smoke] step 3/3 dummy_generate_report  OK
[smoke] PASS  3/3 tool dispatches succeeded.
```

If you see `PASS 3/3` the tool registry, dispatcher, and run isolation are all
working.

---

## Install

The agent framework only needs Python and three lightweight dependencies. The
heavier tool tiers are optional — install only what you plan to use.

| Tier | Env file | Adds | When to install |
| --- | --- | --- | --- |
| Base | `envs/base.yml` | pydicom, SimpleITK, nibabel | Always |
| Inference | `envs/inference.yml` | torch, MONAI | To run segmentation tools |
| Reconstruction | `envs/recon.yml` | pygrappa, h5py | To process raw cardiac k-space |
| Radiomics | `envs/radiomics.yml` | pyradiomics, scikit-image | For ROI feature extraction |

Tool dispatch is controlled by `BCER_TOOL_DISPATCH`:

- `inprocess` (default) — every tool runs in the current Python process.
- `auto` — tier 1/2/3 tools run as subprocesses in their own conda env; base tools stay in process.
- `subprocess` — every tool runs as a subprocess, for maximum isolation.

See [`docs/TOOL_ENV_ANALYSIS.md`](docs/TOOL_ENV_ANALYSIS.md) for the architecture rationale.

---

## Benchmark

The benchmark exercises one `(task, arm, fault)` cell per run.

**Controller arms.** Each arm is a different planning/recovery strategy:

| Paper label | CLI flag |
| --- | --- |
| BCER | `--arm bcer` (alias for `bcer_sketch`) |
| ReAct | `--arm react` |
| ReAct + symbolic binding | `--arm react_token` |
| ReAct + binding + bounded reflector | `--arm react_token_reflector` |

**Fault types.** Injected at most once per run:

| Group | Faults | Scored by |
| --- | --- | --- |
| Recoverable, deterministic | `token_mutation`, `path_mutation` | ERR |
| Recoverable, semantic | `argument_omission`, `semantic_swap`, `space_mismatch` | ERR |
| Nonrecoverable | `missing_modality`, `scope_violation`, `timeout` | safe-halt rate |

Run one task/arm/fault cell against a manifest you built locally:

```bash
python benchmark/benchmark_runner.py \
    --manifest benchmark/cases_manifest.jsonl \
    --task long_prostate_full \
    --arm bcer \
    --fault none \
    --runs-root runs
```

See [`benchmark/README.md`](benchmark/README.md) and
[`docs/METRICS.md`](docs/METRICS.md) for the full CLI and metric definitions.

---

## Datasets

BCER does not ship any medical data. We evaluated the framework on the
following **public datasets** — links and access conditions:

| Domain | Dataset | Access |
| --- | --- | --- |
| Prostate | **fastMRI Prostate** | https://fastmri.med.nyu.edu — release agreement; NYU/Meta |
| Brain | **BraTS 2021** (RSNA-ASNR-MICCAI) | https://www.synapse.org/Synapse:syn25829067 — Synapse account |
| Cardiac (cine) | **ACDC** | https://www.creatis.insa-lyon.fr/Challenge/acdc — registration required |
| Cardiac (raw k-space) | **CMRxRecon 2025** | https://cmrxrecon.github.io — challenge release |

Once you have one or more datasets locally:

```bash
export BCER_PROSTATE_ROOT=/path/to/fastMRI_prostate
export BCER_BRATS_ROOT=/path/to/brats2021
export BCER_ACDC_ROOT=/path/to/acdc
export BCER_CARDIAC_RAW_ROOT=/path/to/cmrxrecon   # optional

python scripts/manifest_builder.py \
    --prostate-root "$BCER_PROSTATE_ROOT" \
    --brain-root    "$BCER_BRATS_ROOT" \
    --cardiac-root  "$BCER_ACDC_ROOT" \
    --output benchmark/cases_manifest.jsonl
```

The manifest builder is layout-tolerant: it scans for NIfTI / DICOM / HDF5 and
infers modalities from filenames and DICOM headers. See
[`docs/DATASETS.md`](docs/DATASETS.md) for the expected directory layout per
domain and how to bring a non-standard dataset.

---

## Documentation

| Document | Audience |
| --- | --- |
| [`docs/TOOLS.md`](docs/TOOLS.md) | Tool call convention, the 21 registered tools, how to add new ones |
| [`docs/METRICS.md`](docs/METRICS.md) | SR / TCR / ERR / safe-halt definitions and how to read a result |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Supported data layouts and manifest format |
| [`docs/TOOL_ENV_ANALYSIS.md`](docs/TOOL_ENV_ANALYSIS.md) | Environment tiering and subprocess dispatch architecture |
| [`benchmark/README.md`](benchmark/README.md) | Benchmark CLI and outputs |
| [`docs/cardiac_acdc_classification_rules.md`](docs/cardiac_acdc_classification_rules.md) | Rule-based cardiac classifier |

---

## Project layout

```
agent/             planner, sketch compiler, executor, reflector, rule engine
benchmark/         runner, summariser, smoke benchmark, paper-arm definitions
commands/          tool registry, dispatcher, schema validation
core/              project paths, parser, plan DAG, domain config
llm/               LLM backend adapters (OpenAI, Anthropic, Gemini, vLLM)
mri_agent_shell/   interactive CLI shell + cerebellum runtime + dummy tools
runtime/           memory, finalisation, artifact index, sandbox
tools/             21 imaging tool wrappers + subprocess entry point
scripts/           manifest builder and one-off utilities
envs/              tiered conda env files
docs/              user-facing documentation
configs/           task contracts and tool runtime tier config
```

---

## Status

BCER is a **research framework**, not a clinically validated product. It is
intended for studying the reliability of agent execution on multi-step
imaging workflows. It does not replace expert radiological review.

The repository tracks the v3 paper code path. Issues and pull requests are
welcome.

## License

MIT — see [`LICENSE`](LICENSE).
