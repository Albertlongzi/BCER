from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from commands.schemas import ToolContext
from tools.rag_search import rag_search_tool


class RagSearchToolTests(unittest.TestCase):
    def test_rag_search_reads_identify_outputs_and_answers_question(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            run_dir = ws / "runs" / "case_x" / "run_1"
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            ingest_dir = artifacts_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)
            dicom_meta = ingest_dir / "dicom_meta.json"
            dicom_meta.write_text(
                json.dumps(
                    {
                        "series": [
                            {"sequence_guess": "DWI", "TR": 4200, "TE": 87},
                        ]
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            case_state_path = run_dir / "case_state.json"
            case_state = {
                "case_id": "case_x",
                "run_id": "run_1",
                "stage_outputs": {
                    "identify": {
                        "identify_sequences": [
                            {
                                "call_id": "c1",
                                "ok": True,
                                "data": {
                                    "dicom_meta_path": str(dicom_meta),
                                },
                            }
                        ]
                    }
                },
            }
            case_state_path.write_text(json.dumps(case_state, indent=2) + "\n", encoding="utf-8")

            ctx = ToolContext(
                case_id="case_x",
                run_id="run_1",
                run_dir=run_dir,
                artifacts_dir=artifacts_dir,
                case_state_path=case_state_path,
            )
            out = rag_search_tool(
                args={"question": "What is the TR and TE of the DWI sequence?"},
                ctx=ctx,
            )
            data = dict(out.get("data") or {})
            self.assertTrue(bool(str(data.get("answer") or "").strip()))
            self.assertTrue(len(list(data.get("evidence_snippets") or [])) > 0)
            self.assertTrue(Path(str(data.get("results_path") or "")).exists())

    def test_rag_search_follows_index_references_to_header_txt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            run_dir = ws / "runs" / "case_x" / "run_2"
            artifacts_dir = run_dir / "artifacts"
            ingest_dir = artifacts_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            index_txt = ingest_dir / "series_index.txt"
            header_txt = ingest_dir / "t2wsfov_header.txt"
            index_txt.write_text("header_file=t2wsfov_header.txt\n", encoding="utf-8")
            header_txt.write_text("EchoTime: 90\nRepetitionTime: 4200\n", encoding="utf-8")

            case_state_path = run_dir / "case_state.json"
            case_state = {
                "case_id": "case_x",
                "run_id": "run_2",
                "stage_outputs": {
                    "identify": {
                        "identify_sequences": [
                            {
                                "call_id": "c2",
                                "ok": True,
                                "data": {
                                    "series_index_path": str(index_txt),
                                },
                            }
                        ]
                    }
                },
            }
            case_state_path.write_text(json.dumps(case_state, indent=2) + "\n", encoding="utf-8")

            ctx = ToolContext(
                case_id="case_x",
                run_id="run_2",
                run_dir=run_dir,
                artifacts_dir=artifacts_dir,
                case_state_path=case_state_path,
            )
            out = rag_search_tool(
                args={"question": "What is the TE value in the header?"},
                ctx=ctx,
            )
            data = dict(out.get("data") or {})
            snippets = list(data.get("evidence_snippets") or [])
            self.assertTrue(any("t2wsfov_header.txt" in str(s.get("path") or "") for s in snippets))

    def test_rag_search_scans_large_referenced_header_for_te(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            run_dir = ws / "runs" / "case_x" / "run_3"
            artifacts_dir = run_dir / "artifacts"
            ingest_dir = artifacts_dir / "ingest"
            header_dir = ingest_dir / "dicom_txt"
            header_dir.mkdir(parents=True, exist_ok=True)

            header_txt = header_dir / "t2wsfov_header.txt"
            filler = "\n".join([f"filler_line_{i}: value_{i}" for i in range(2500)])
            header_txt.write_text(f"{filler}\n(0018,0081) EchoTime: 132\nRepetitionTime: 4000\n", encoding="utf-8")
            index_txt = ingest_dir / "dicom_headers_index.txt"
            index_txt.write_text("t2wsfov\tdicom_txt/t2wsfov_header.txt\n", encoding="utf-8")

            case_state_path = run_dir / "case_state.json"
            case_state = {
                "case_id": "case_x",
                "run_id": "run_3",
                "stage_outputs": {
                    "identify": {
                        "identify_sequences": [
                            {
                                "call_id": "c3",
                                "ok": True,
                                "data": {
                                    "dicom_headers_index_path": str(index_txt),
                                },
                            }
                        ]
                    }
                },
            }
            case_state_path.write_text(json.dumps(case_state, indent=2) + "\n", encoding="utf-8")

            ctx = ToolContext(
                case_id="case_x",
                run_id="run_3",
                run_dir=run_dir,
                artifacts_dir=artifacts_dir,
                case_state_path=case_state_path,
            )
            out = rag_search_tool(
                args={"question": "What is the Echo Time (TE) of T2w?"},
                ctx=ctx,
            )
            data = dict(out.get("data") or {})
            snippets = list(data.get("evidence_snippets") or [])
            self.assertTrue(any("t2wsfov_header.txt" in str(s.get("path") or "") for s in snippets))
            self.assertTrue(any("EchoTime" in str(s.get("line") or "") for s in snippets))


if __name__ == "__main__":
    unittest.main()
