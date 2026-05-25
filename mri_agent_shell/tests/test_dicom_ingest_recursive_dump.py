from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from commands.schemas import ToolContext
from tools.dicom_ingest import identify_sequences

try:
    import pydicom  # type: ignore
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset  # type: ignore
    from pydicom.sequence import Sequence  # type: ignore
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid  # type: ignore

    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False


@unittest.skipUnless(_HAS_PYDICOM, "pydicom is required for recursive DICOM dump test")
class DicomIngestRecursiveDumpTests(unittest.TestCase):
    def _write_synthetic_dicom(self, path: Path) -> None:
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.Modality = "MR"
        ds.SeriesDescription = "t2wsfov"
        ds.ProtocolName = "T2W"
        ds.EchoTime = 90
        ds.RepetitionTime = 4200
        ds.Rows = 2
        ds.Columns = 2
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = b"\x00" * 8

        diff_item = Dataset()
        diff_item.DiffusionBValue = 1400
        ds.MRDiffusionSequence = Sequence([diff_item])

        ds.save_as(str(path), write_like_original=False)

    def test_identify_sequences_writes_recursive_unrolled_header_dump(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            series_dir = ws / "case_x" / "t2wsfov"
            series_dir.mkdir(parents=True, exist_ok=True)
            dcm_path = series_dir / "img0001.dcm"
            self._write_synthetic_dicom(dcm_path)

            run_dir = ws / "runs" / "case_x" / "run_1"
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            case_state_path = run_dir / "case_state.json"
            case_state_path.write_text("{\"case_id\":\"case_x\",\"run_id\":\"run_1\"}\n", encoding="utf-8")

            ctx = ToolContext(
                case_id="case_x",
                run_id="run_1",
                run_dir=run_dir,
                artifacts_dir=artifacts_dir,
                case_state_path=case_state_path,
            )

            out = identify_sequences(
                args={
                    "dicom_case_dir": str(ws / "case_x"),
                    "convert_to_nifti": False,
                    "output_subdir": "ingest",
                    "require_pydicom": True,
                },
                ctx=ctx,
            )
            data = dict(out.get("data") or {})
            index_path = Path(str(data.get("dicom_headers_index_path") or ""))
            self.assertTrue(index_path.exists())
            lines = [x for x in index_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertTrue(lines)
            header_txt = Path(lines[0].split("\t", 1)[1])
            self.assertTrue(header_txt.exists())

            txt = header_txt.read_text(encoding="utf-8", errors="ignore")
            self.assertIn("=== KEY_PHYSICAL_PARAMETERS ===", txt)
            self.assertIn("EchoTime: 90", txt)
            self.assertIn("RepetitionTime: 4200", txt)
            self.assertIn("=== DICOM_RECURSIVE_DUMP", txt)
            self.assertIn("(0018,0081) EchoTime: 90", txt)
            self.assertIn("MRDiffusionSequence", txt)
            self.assertIn("DiffusionBValue: 1400", txt)


if __name__ == "__main__":
    unittest.main()


def test_identify_sequences_h5_file_input_does_not_raise_not_a_directory() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        h5_path = ws / "cardiac_Center001_UIH_30T_umr780_P002_cine_sax.h5"
        h5_path.write_bytes(b"fakeh5")

        run_dir = ws / "runs" / "case_x" / "run_1"
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        case_state_path = run_dir / "case_state.json"
        case_state_path.write_text("{\"case_id\":\"case_x\",\"run_id\":\"run_1\"}\n", encoding="utf-8")

        ctx = ToolContext(
            case_id="case_x",
            run_id="run_1",
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            case_state_path=case_state_path,
        )

        out = identify_sequences(
            args={
                "dicom_case_dir": str(h5_path),
                "convert_to_nifti": False,
                "output_subdir": "ingest",
                "require_pydicom": False,
            },
            ctx=ctx,
        )
        data = dict(out.get("data") or {})
        assert isinstance(data.get("mapping"), dict)
        # H5 file-case should not crash; mapping may be empty or heuristic CINE depending on filename.
        assert "Not a directory" not in " ".join(str(w) for w in (out.get("warnings") or []))
