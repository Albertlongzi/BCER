# BCER Paper Benchmark

This benchmark exposes the paper-facing controller modes only:

| Paper label | CLI arm | Internal runtime |
| --- | --- | --- |
| BCER | `bcer` or `bcer_sketch` | constrained sketch planner + compiler + Cerebellum + bounded reflector |
| ReAct | `react` | direct reactive tool calls |
| ReAct+Bind | `react_token` | reactive calls with symbolic binding |
| ReAct+Bind+Ref | `react_token_reflector` | reactive calls with binding and bounded reflection |

## Run

Generate a manifest from your local dataset roots with `scripts/manifest_builder.py`,
then run one task/arm:

```bash
python benchmark/benchmark_runner.py \
  --manifest benchmark/cases_manifest.jsonl \
  --task long_prostate_full \
  --arm bcer \
  --fault none \
  --runs-root runs
```

Build the manifest from local dataset roots (no paths are baked into the repo):

```bash
python scripts/manifest_builder.py \
  --prostate-root "$BCER_PROSTATE_ROOT" \
  --brain-root "$BCER_BRATS_ROOT" \
  --cardiac-root "$BCER_ACDC_ROOT" \
  --output benchmark/cases_manifest.jsonl
```

## Metrics

The runner records:

- `SR`: binary case-level task-contract success.
- `TCR`: fraction of required contract milestones/artifacts completed and validated.
- `ERR`: recovery rate for recoverable injected faults.
- `safe_halt_rate`: correct halt/block behavior for nonrecoverable faults.

See `docs/METRICS.md` for full definitions.

## Tool Dispatch

By default tools run in process:

```bash
BCER_TOOL_DISPATCH=inprocess python benchmark/benchmark_runner.py ...
```

To use tiered conda subprocess dispatch:

```bash
BCER_TOOL_DISPATCH=auto python benchmark/benchmark_runner.py ...
```

The tier configuration is `configs/tool_runtime.yml`; env definitions are under
`envs/`. Heavy tool assets/checkpoints should live under `assets/`, not outside
the project.

## Summaries

```bash
python benchmark/summarize_results.py --results-dir benchmark --mode baseline
python benchmark/summarize_results.py --results-dir benchmark --mode detail --arm bcer
```
