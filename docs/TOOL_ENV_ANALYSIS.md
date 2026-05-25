# Tool Environment Architecture Analysis

**Context:** BCER_open — agentic MRI analysis framework for paper release.
This document analyses the current tool environment setup, its pain points, and
concrete options for isolating tool environments, especially for tools that are
ML model inference pipelines rather than plain scripts.

---

## 1. Current State

All 24 tools share a single flat Python environment defined by `requirements.txt`:

```
pydicom, SimpleITK, numpy, pillow          # DICOM / image I/O
torch>=2.1, monai>=1.3, nibabel            # GPU segmentation
scikit-image, pyradiomics                  # radiomics features
pygrappa, h5py                             # k-space reconstruction  (implied)
bm3d                                       # classical denoising     (implied)
openai, anthropic, google-generativeai     # cloud LLM adapters
PyYAML, jsonschema                         # agent runtime
```

The agent loop (`agent/loop.py`) builds a `ToolRegistry`, then the dispatcher
(`commands/dispatcher.py`) validates JSON arguments and calls the tool function
**in-process** — same Python interpreter, same GPU context, same memory space.

---

## 2. Why This Hurts

### 2.1 Dependency conflicts
`pyradiomics` and `pygrappa` both carry compiled C extensions that are sensitive
to the exact NumPy/SciPy ABI version. Pinning `numpy<2` (required by monai 1.3)
is already causing friction; adding radiomics and reconstruction to the same env
makes version negotiation fragile and hard to replicate for other researchers.

### 2.2 In-process model loading is expensive and wasteful
MONAI bundles (brain BraTS, prostate anatomy) take **30–60 s to load** from disk.
In the current architecture, every benchmark run that starts a fresh process pays
this cost even for tasks that don't use segmentation. There is also no shared model
cache across the agent's tool calls; each call site re-invokes the loading logic.

### 2.3 GPU memory fights
If `brain_tumor_segmentation` and `cardiac_cine_segmentation` are both registered
in the same process and a multi-domain agent triggers them in the same session, both
MONAI networks will compete for VRAM simultaneously. The current design has no
mechanism to sequence or cap GPU allocation.

### 2.4 Single point of failure
A segfault inside `pygrappa` or a CUDA OOM inside a segmentation model will crash
the whole agent process, losing all intermediate results in `CaseState`.

### 2.5 Conceptual mismatch
Plain-script tools (`dicom_ingest`, `alignment_qc`, `bm3d_denoising`) are pure
functions over files. ML-inference tools (`brain_tumor_segmentation`,
`prostate_segmentation`, `cardiac_cine_segmentation`) are **stateful services**:
they load a large model, run inference, and ideally keep the model in memory for
subsequent calls. Mixing both patterns into the same JSON-dispatch interface
obscures this difference.

---

## 3. Tool Taxonomy by Environment

Group tools by their actual dependency footprint rather than their pipeline stage:

| Tier | Name          | Tools                                                                                                      | Key Extra Deps                     |
|------|---------------|------------------------------------------------------------------------------------------------------------|------------------------------------|
| 0    | **Base**      | dicom_ingest, dicom_paths, alignment_qc, materialize_registration, registration, resample_image, compare_nifti_slices, generate_qa_snapshot, report_generation, vlm_evidence, sandbox_exec, rag_search | pydicom, SimpleITK, nibabel, numpy |
| 1    | **Inference** | prostate_segmentation, brain_tumor_segmentation, cardiac_cine_segmentation, prostate_lesion_candidates, prostate_distortion_correction | + torch, monai                     |
| 2    | **Recon**     | reconstruct_grappa                                                                                         | + pygrappa, h5py                   |
| 3    | **Radiomics** | roi_features                                                                                               | + pyradiomics, scikit-image        |
| 4    | **Classify**  | brain_glioma_grade_classification, cardiac_cine_classification                                             | (Tier 0 deps only — rule-based)    |

A few observations:
- Tier 4 tools need no extra deps despite being "model-like" in name — they are
  pure rule engines that read mask files. They belong in Tier 0.
- `bm3d_denoising` needs only the `bm3d` package, which installs cleanly
  alongside numpy. It fits in Tier 0 with one extra pip install.
- The agent orchestrator itself (llm adapters, planner, reflector) only needs
  Tier 0 + LLM API clients. It does **not** need torch at all.

---

## 4. Options

### Option A — Tiered conda environments + subprocess dispatch
*Recommended for paper release code.*

Create 3–4 environment files:

```
envs/
  base.yml        # Tier 0 + Tier 4 + agent runtime
  inference.yml   # base + torch + monai
  recon.yml       # base + pygrappa + h5py
  radiomics.yml   # base + pyradiomics + scikit-image
```

The agent orchestrator runs in `base`. When it dispatches a Tier 1/2/3 tool it
calls the tool as a **subprocess** using that tier's conda env:

```python
conda run -n inference python -m bcer.tools.brain_tumor_segmentation \
    --args-file /tmp/run-123/args.json \
    --result-file /tmp/run-123/result.json
```

Each tool already has well-defined `ToolSpec` schemas and `arg_models`. Exposing a
`--args-file / --result-file` CLI entrypoint on each Tier 1–3 tool is a small,
mechanical change.

**Pros:**
- Clean environment isolation per dependency group.
- Zero protocol overhead — standard subprocess + JSON files.
- Reproducible: researchers can install only the env they need.
- Crash in subprocess cannot corrupt the agent process.

**Cons:**
- Subprocess startup adds ~1–2 s per tool call (process spawn + Python startup).
- Model still reloads on every call unless you add a worker pool (see Option B).
- Managing 4 conda environments adds some ops overhead.

**Verdict:** Best fit for the paper release goal. The repo ships with clear
`envs/*.yml` files; reviewers can install exactly what they need. The subprocess
interface requires ~50 lines of glue code in the dispatcher.

---

### Option B — Persistent worker processes for inference tools
*Best if you keep developing this after the paper.*

Run each Tier 1 tool as a **long-lived worker process** that loads its MONAI bundle
once and accepts requests over a lightweight channel (e.g., Unix domain socket or
shared memory queue). The agent dispatcher sends a JSON job and reads back a JSON
result without re-loading the model.

A minimal version looks like this:

```
agent process   ──JSON job──►   segmentation_worker.py
                ◄─JSON result──   (MONAI bundle loaded once at startup)
```

You do not need a full HTTP server or MCP for this. Python's `multiprocessing`
`Queue`, a named pipe, or even a file-based "inbox/outbox" per worker is enough.
Workers live in their own conda env (same as Option A) launched by the agent at
startup or on first use.

**Pros:**
- Model loading cost paid once, amortized across all tool calls in a session.
  Brain BraTS bundle: 45 s first call, ~0.5 s subsequent calls.
- GPU memory is explicitly owned by one process; no resource fights.
- Worker crashes are isolated; the agent can restart the worker and retry.

**Cons:**
- Worker lifecycle management (start, health-check, graceful shutdown).
- Adds ~100–150 lines of infrastructure code.
- Harder to debug than a plain subprocess call.

**Verdict:** The right next step once the paper version is stable. The amortized
model loading alone makes this worthwhile for interactive or benchmark use.

---

### Option C — Lazy imports + optional dependency groups (minimal change)
*Good enough for paper release if you want zero architecture change.*

Keep the monolithic env but split `requirements.txt` into layered files:

```
requirements-base.txt      # pydicom, SimpleITK, nibabel, numpy, PyYAML, jsonschema
requirements-inference.txt # -r base + torch, monai
requirements-recon.txt     # -r base + pygrappa, h5py
requirements-radiomics.txt # -r base + pyradiomics, scikit-image
requirements-all.txt       # -r inference + recon + radiomics + LLM APIs
```

In each Tier 1–3 tool, wrap the heavy import in a `try/except ImportError` and
raise a clear `ToolUnavailableError` if the package is not installed. The agent
then reports "tool not available in this environment" rather than crashing.

**Pros:**
- Zero architecture change.
- A researcher who only wants to run brain segmentation installs only
  `requirements-inference.txt`.
- Graceful degradation for optional tools (radiomics, GRAPPA).

**Cons:**
- No process isolation — a segfault or OOM still kills the whole agent.
- Model loading overhead unchanged.
- Dependency conflicts within the full env remain.

**Verdict:** A good first step that can be done in a day and ships with the paper.
Combine with Option A (subprocess dispatch) for the isolation piece.

---

### Option D — MCP servers
*Most principled long-term; most complex short-term.*

Each tool (or tier) becomes an MCP server. The agent uses the MCP client protocol
to call tools across process boundaries, with full schema introspection and typed
results. This is essentially Option B with a standardised wire protocol instead of
a custom channel.

**Pros:**
- Standard protocol; integrates with Claude Code and other MCP-aware agents.
- Full process isolation, model persistence, schema-first.
- Each server can live in its own Docker container for full reproducibility.

**Cons:**
- Non-trivial refactor of the entire tool-dispatch layer.
- Adds a protocol layer that every contributor must understand.
- Heavyweight relative to what the codebase actually needs today.

**Verdict:** Worth considering for a v4 productionized workstation, not for the
paper release. The complexity cost is too high given the current development stage.

---

## 5. Recommendation

For **BCER_open (paper release)**, apply in order:

1. **Split requirements.txt** into `base`, `inference`, `recon`, `radiomics` files
   (Option C). This is a one-hour change that immediately improves reproducibility
   and documents the dependency tiers explicitly.

2. **Add subprocess dispatch for Tier 1 tools** (Option A, inference tier only).
   The three MONAI segmentation tools are the ones most likely to OOM, segfault,
   or conflict with each other. Dispatching them as subprocesses via
   `conda run -n inference ...` isolates the risk without touching the agent
   architecture.

3. **In-process model cache** (no architecture change). For the case where
   subprocess dispatch is not set up yet, add a module-level dict
   `_LOADED_BUNDLES: dict[str, Any]` in each segmentation tool. On repeat calls
   within the same process, return the cached network instead of re-loading.
   Drops segmentation latency from ~50 s to ~0.5 s for all calls after the first.

For **post-paper development**, add the persistent worker model (Option B) for the
inference tier. The subprocess interface from step 2 above is already the right I/O
contract — upgrading from "one subprocess per call" to "one long-lived worker" is
mostly an infrastructure change with no tool logic change.

---

## 6. What NOT to Do

- **One conda env per tool** — overkill. There are only 4 real dependency clusters.
  24 environments would create more management overhead than they solve.
- **Full MCP now** — too much refactor for a paper release. The payoff is real but
  the timing is wrong.
- **Docker per tool** — appropriate for a deployed service, not a research codebase
  that researchers need to install and modify locally.
- **Ignore this entirely** — the current setup works for a single researcher's
  machine where all deps happen to coexist, but will frustrate reviewers and future
  contributors who only need a subset of the pipeline.

---

## 7. File Changes Summary

If you go with Recommendation steps 1–2:

```
requirements.txt           → keep for "install everything" convenience
requirements-base.txt      → new: Tier 0 deps + agent runtime
requirements-inference.txt → new: base + torch + monai
requirements-recon.txt     → new: base + pygrappa + h5py
requirements-radiomics.txt → new: base + pyradiomics + scikit-image
envs/inference.yml         → new: conda env spec for Tier 1 workers
commands/dispatcher.py     → add ~50 lines: subprocess dispatch for Tier 1 tools
tools/brain_tumor_segmentation.py  → add __main__ block (args-file → result-file)
tools/prostate_segmentation.py     → same
tools/cardiac_cine_segmentation.py → same
```

The agent loop, reflector, planner, and all Tier 0 tools are untouched.
