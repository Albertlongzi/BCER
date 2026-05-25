#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_import_path() -> None:
    here = Path(__file__).resolve()
    ws_root = here.parents[2]
    if str(ws_root) not in sys.path:
        sys.path.insert(0, str(ws_root))


def main() -> None:
    _ensure_import_path()
    from commands.schemas import ToolContext  # noqa: WPS433
    from tools.cardiac_cine_classification import classify_cardiac_cine_disease  # noqa: WPS433

    ap = argparse.ArgumentParser(description="Run rule-based cardiac cine disease classification.")
    ap.add_argument("--seg-path", required=True, help="Cardiac segmentation NIfTI path (labels: 1=RV,2=MYO,3=LV).")
    ap.add_argument("--cine-path", default="", help="Optional cine NIfTI path.")
    ap.add_argument("--patient-info-path", default="", help="Optional Info.cfg path (ED/ES/Group/Height/Weight).")
    ap.add_argument("--ed-frame", type=int, default=None, help="Optional ED frame index (1-based).")
    ap.add_argument("--es-frame", type=int, default=None, help="Optional ES frame index (1-based).")
    ap.add_argument("--output-dir", required=True, help="Output root directory.")
    ap.add_argument("--output-subdir", default="classification/cardiac_cine", help="Subdirectory under output-dir.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = ToolContext(
        case_id="cardiac_demo",
        run_id="manual",
        run_dir=out_dir,
        artifacts_dir=out_dir,
        case_state_path=out_dir / "case_state.json",
    )
    tool_args = {
        "seg_path": str(Path(args.seg_path).expanduser().resolve()),
        "output_subdir": str(args.output_subdir),
    }
    if args.cine_path:
        tool_args["cine_path"] = str(Path(args.cine_path).expanduser().resolve())
    if args.patient_info_path:
        tool_args["patient_info_path"] = str(Path(args.patient_info_path).expanduser().resolve())
    if args.ed_frame is not None:
        tool_args["ed_frame"] = int(args.ed_frame)
    if args.es_frame is not None:
        tool_args["es_frame"] = int(args.es_frame)

    result = classify_cardiac_cine_disease(tool_args, ctx)
    data = result.get("data", {}) if isinstance(result, dict) else {}

    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonable(x) for x in obj]
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return _jsonable(obj.to_dict())
            except Exception:
                return str(obj)
        return obj

    summary_json = out_dir / "cardiac_cine_classification_result.json"
    summary_json.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Result JSON: {summary_json}")
    print(f"[OK] Predicted group: {data.get('predicted_group')}")
    print(f"[OK] Classification path: {data.get('classification_path')}")


if __name__ == "__main__":
    main()
