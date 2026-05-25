# Tools

BCER tools are deterministic Python functions wrapped with a strict JSON
input/output schema. The agent loop, the benchmark runner, and end users all
call them through the same `ToolDispatcher`. This document covers the call
convention, the registered tools, and how to add a new one.

---

## 1. The call convention

Every tool consists of three things:

- **A `ToolSpec`** declaring the tool name, description, input schema, and
  output schema (`commands/schemas.py:ToolSpec`).
- **A Python function** of signature `(args: dict, ctx: ToolContext) -> dict`.
- **A `Tool(spec, func)` registration** added to the `ToolRegistry`.

The `ToolDispatcher` validates arguments against the input schema, runs the
function, validates outputs against the output schema, records an
`ExecutionLogEntry`, and returns a `ToolResult` with `ok`, `data`, `artifacts`,
`warnings`, and `error` fields.

Minimal end-to-end example (this is what `benchmark/smoke.py` does):

```python
from pathlib import Path
from commands.dispatcher import ToolDispatcher
from commands.registry import ToolRegistry
from commands.schemas import ToolCall
from mri_agent_shell.dummy_tools import build_dummy_tools

reg = ToolRegistry()
for tool in build_dummy_tools():
    reg.register(tool)

dispatcher = ToolDispatcher(registry=reg, runs_root=Path("/tmp/bcer_demo"))
state, ctx = dispatcher.create_run(case_id="demo", run_id="run_001")

call = ToolCall(
    tool_name="dummy_segment",
    arguments={"case_path": "/tmp/case", "anatomy": "demo"},
    call_id="c1",
    case_id=state.case_id,
)
result = dispatcher.dispatch(call, state, ctx)
print(result.ok, result.artifacts)
```

The dispatcher writes all artifacts under
`<runs_root>/<case_id>/<run_id>/artifacts/<tool_output_subdir>/` and an
append-only `execution_log.jsonl` per run.

---

## 2. Registered tools

All tools are registered in `tools/catalog.py:build_registry`.

### Always available (Tier 0 — base env)

These tools only need `pydicom`, `SimpleITK`, `nibabel`, `numpy`, `pillow`, and
work CPU-only.

| Tool | Purpose |
| --- | --- |
| `identify_sequences` | DICOM header inventory; map to T2w / ADC / DWI / T1c / FLAIR; optional NIfTI conversion |
| `register_to_reference` | SimpleITK resample of a moving volume into a fixed reference space |
| `alignment_qc` | Quick alignment QC: affine consistency, spacing match, overlay slices |
| `materialize_registration` | Materialize registration outputs without resampling (when geometry already matches) |
| `resample_image` | Resample a NIfTI volume to a target physical grid without alignment optimisation |
| `compare_nifti_slices` | Side-by-side centre-slice comparison PNG of two NIfTI volumes |
| `generate_qa_snapshot` | Grayscale PNG of a representative centre slice from a NIfTI volume |
| `classify_brain_glioma_grade` | Rule-based HGG / LGG glioma grade from ROI feature CSV |
| `classify_cardiac_cine_disease` | Rule-based NOR / MINF / DCM / HCM / RV cardiac classification from segmentation |
| `generate_report` | Render a case report (Markdown + JSON) from `case_state.json` and existing artifacts |
| `package_vlm_evidence` | Bundle artifact paths and summary into a `vlm_evidence_bundle.json` |
| `denoise_image_bm3d` | Classical 2D-slicewise BM3D denoising on a NIfTI volume |
| `rag_search` | Local question answering over text/JSON artifacts in a run directory |
| `sandbox_exec` | Run a shell command inside a tool-scoped artifact directory |

### Inference (Tier 1 — `envs/inference.yml`)

Need `torch` + `monai`. GPU recommended.

| Tool | Purpose |
| --- | --- |
| `segment_prostate` | MONAI prostate zonal segmentation on T2w (CG/PZ + whole-gland mask) |
| `brats_mri_segmentation` | MONAI BraTS tumour subregions (TC / WT / ET) on aligned 1 mm brain MRI |
| `segment_cardiac_cine` | Cardiac cine LV/RV/myocardium segmentation (per-frame) |
| `detect_lesion_candidates` | Prostate mpMRI lesion candidate detection from registered T2w + ADC + DWI |
| `correct_prostate_distortion` | EPI/diffusion distortion correction (external diffusion backend) |

### Reconstruction (Tier 2 — `envs/recon.yml`)

Needs `pygrappa` + `h5py`.

| Tool | Purpose |
| --- | --- |
| `reconstruct_grappa` | k-space GRAPPA reconstruction or direct H5 image → NIfTI |

### Radiomics (Tier 3 — `envs/radiomics.yml`)

Needs `pyradiomics` + `scikit-image`.

| Tool | Purpose |
| --- | --- |
| `extract_roi_features` | Per-ROI × per-sequence radiomics + texture feature CSV |

See `docs/TOOL_ENV_ANALYSIS.md` for the rationale behind the tiering and the
subprocess dispatch design.

---

## 3. Dispatch modes

`BCER_TOOL_DISPATCH` controls how tools are executed:

| Mode | Behaviour | When to use |
| --- | --- | --- |
| `inprocess` (default) | All tools run in the current Python process | Single env install; tests; the smoke benchmark |
| `auto` | Tier 1 / 2 / 3 tools run as subprocesses in their conda env; Tier 0 stays in process | Multi-tier install where the agent runtime is light and inference is heavy |
| `subprocess` | Every tool runs as a subprocess | Maximum isolation for debugging |

The tier configuration is in `configs/tool_runtime.yml`. The subprocess entry
point is `tools/run_tool.py`.

---

## 4. Adding a new tool

A new tool is one file in `tools/` plus a one-line addition to
`tools/catalog.py`. Here is a complete minimal example:

```python
# tools/my_tool.py
from typing import Any, Dict
from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec

MY_TOOL_SPEC = ToolSpec(
    name="my_tool",
    description="Write a greeting to a text file in the artifacts directory.",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "output_subdir": {"type": "string"},
        },
        "required": ["name"],
    },
    output_schema={
        "type": "object",
        "properties": {"greeting_path": {"type": "string"}},
        "required": ["greeting_path"],
    },
    version="0.1.0",
    tags=["demo"],
)


def _my_tool(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    out_dir = ctx.artifacts_dir / str(args.get("output_subdir") or "my_tool")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "greeting.txt"
    out_path.write_text(f"Hello, {args['name']}!\n", encoding="utf-8")

    artifact = ArtifactRef(path=str(out_path), kind="text", description="Greeting")
    return {
        "data": {"greeting_path": str(out_path)},
        "artifacts": [artifact],
        "generated_artifacts": [artifact],
        "warnings": [],
    }


def build_tool() -> Tool:
    return Tool(spec=MY_TOOL_SPEC, func=_my_tool)
```

Then register it in `tools/catalog.py`:

```python
from tools.my_tool import build_tool as build_my_tool
# ... inside build_registry():
reg.register(build_my_tool())
```

The tool will now be visible to the agent loop, the benchmark runner, and
subprocess dispatch. No other code changes are required.

### Conventions worth following

- Always write artifacts under `ctx.artifacts_dir / <output_subdir>` so the run
  directory stays self-contained.
- Always return at least `data`, `artifacts`, `generated_artifacts`, and
  `warnings` in the dict. The dispatcher tolerates extra keys.
- Use `ArtifactRef.kind` consistently: `"nifti"`, `"dicom"`, `"json"`, `"csv"`,
  `"text"`, `"figure"`, `"transform"`.
- Keep tools deterministic: same inputs → same outputs. Non-determinism
  (random seeds, sampling) should be controlled by an explicit argument.
- Validate file existence with clear error messages, not silent skips. The
  reflector relies on actionable error messages to recover.

---

## 5. Where to look in the code

- `commands/schemas.py` — `ToolSpec`, `ToolCall`, `ToolResult`, `ArtifactRef`, `ToolContext`
- `commands/registry.py` — `ToolRegistry`
- `commands/dispatcher.py` — `ToolDispatcher` (validation, logging, domain whitelist, subprocess dispatch)
- `commands/tool_runtime.py` — `ToolRuntimeConfig` (per-tier conda env config)
- `tools/run_tool.py` — subprocess entry point invoked when a tool is dispatched in a different env
- `tools/catalog.py` — single source of truth for what tools the agent sees
- `mri_agent_shell/dummy_tools.py` — three minimal example tools used by `benchmark/smoke.py`
