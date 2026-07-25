"""Mask-centred framing in generate_qa_snapshot.

The defect these cover: on a full cardiac field of view the heart is a few dozen
pixels, so a full-FOV overlay reads as "a ring roughly in the right place"
rather than a reviewable segmentation.  When a mask is supplied the preview is
cropped to the labelled region plus a margin -- and it must fall back to the
old full-FOV behaviour whenever there is nothing to crop to.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from commands.schemas import ToolContext
from tools.generate_qa_snapshot import compute_mask_crop_box

_HAS_MPL = importlib.util.find_spec("matplotlib") is not None
_HAS_NIB = importlib.util.find_spec("nibabel") is not None


class ComputeMaskCropBoxTests(unittest.TestCase):
    def test_returns_none_for_empty_mask(self) -> None:
        mask = np.zeros((208, 162), dtype=np.uint8)
        self.assertIsNone(compute_mask_crop_box(mask, np=np, spacing_xy=(1.9, 2.5)))

    def test_returns_none_for_missing_or_wrong_rank_mask(self) -> None:
        self.assertIsNone(compute_mask_crop_box(None, np=np, spacing_xy=(1.0, 1.0)))
        self.assertIsNone(
            compute_mask_crop_box(np.ones((4, 4, 4), dtype=np.uint8), np=np, spacing_xy=(1.0, 1.0))
        )

    def test_small_central_blob_shrinks_the_frame_and_stays_inside(self) -> None:
        mask = np.zeros((208, 162), dtype=np.uint8)
        mask[96:116, 74:90] = 3
        box = compute_mask_crop_box(mask, np=np, spacing_xy=(1.0, 1.0), margin_frac=0.35)
        self.assertIsNotNone(box)
        x0, x1, y0, y1 = box
        self.assertLess(x1 - x0, 208)
        self.assertLess(y1 - y0, 162)
        # The labelled region is fully contained.
        self.assertLessEqual(x0, 96)
        self.assertGreaterEqual(x1, 116)
        self.assertLessEqual(y0, 74)
        self.assertGreaterEqual(y1, 90)
        # And the labels occupy a far larger share of the cropped frame.
        before = float(np.count_nonzero(mask)) / mask.size
        cropped = mask[x0:x1, y0:y1]
        after = float(np.count_nonzero(cropped)) / cropped.size
        self.assertGreater(after, 8.0 * before)

    def test_box_is_square_in_millimetres_not_in_voxels(self) -> None:
        # 1 mm along x, 4 mm along y: a physically square box needs 4x as many
        # x voxels as y voxels.
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[120:136, 120:136] = 1
        box = compute_mask_crop_box(mask, np=np, spacing_xy=(1.0, 4.0), margin_frac=0.0)
        self.assertIsNotNone(box)
        x0, x1, y0, y1 = box
        width_mm = (x1 - x0) * 1.0
        height_mm = (y1 - y0) * 4.0
        self.assertAlmostEqual(width_mm, height_mm, delta=4.0)

    def test_margin_widens_the_box(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[120:136, 120:136] = 1
        tight = compute_mask_crop_box(mask, np=np, spacing_xy=(1.0, 1.0), margin_frac=0.0)
        padded = compute_mask_crop_box(mask, np=np, spacing_xy=(1.0, 1.0), margin_frac=0.5)
        self.assertIsNotNone(tight)
        self.assertIsNotNone(padded)
        self.assertGreater(padded[1] - padded[0], tight[1] - tight[0])
        self.assertGreater(padded[3] - padded[2], tight[3] - tight[2])

    def test_mask_covering_the_whole_frame_returns_none(self) -> None:
        mask = np.ones((64, 64), dtype=np.uint8)
        self.assertIsNone(compute_mask_crop_box(mask, np=np, spacing_xy=(1.0, 1.0)))

    def test_offcentre_blob_box_is_clamped_into_bounds(self) -> None:
        mask = np.zeros((208, 162), dtype=np.uint8)
        mask[0:12, 150:162] = 2
        box = compute_mask_crop_box(mask, np=np, spacing_xy=(1.0, 1.0), margin_frac=0.5)
        self.assertIsNotNone(box)
        x0, x1, y0, y1 = box
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 208)
        self.assertLessEqual(y1, 162)
        self.assertTrue(bool(np.any(mask[x0:x1, y0:y1] > 0)))


@unittest.skipUnless(_HAS_MPL and _HAS_NIB, "matplotlib and nibabel are required to render")
class QaSnapshotRenderTests(unittest.TestCase):
    """Drives the tool end-to-end on a synthetic 4-D cine + 3-D mask."""

    def setUp(self) -> None:
        import nibabel as nib

        self.tmp = Path(tempfile.mkdtemp(prefix="bcer_qa_crop_"))
        self.ctx = ToolContext(
            case_id="c",
            run_id="r",
            run_dir=self.tmp,
            artifacts_dir=self.tmp / "artifacts",
            case_state_path=self.tmp / "case_state.json",
        )
        self.ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)

        nx, ny, nz, nt = 208, 162, 8, 6
        anat = np.random.RandomState(0).rand(nx, ny, nz, nt).astype(np.float32)
        affine = np.diag([1.923077, 2.492877, 10.0, 1.0])
        self.anat_path = self.tmp / "cine.nii.gz"
        nib.save(nib.Nifti1Image(anat, affine), str(self.anat_path))

        # A heart-sized blob on the centre slice only.
        mask = np.zeros((nx, ny, nz), dtype=np.uint8)
        mask[96:118, 72:92, nz // 2] = 3
        mask[92:122, 68:96, nz // 2][mask[92:122, 68:96, nz // 2] == 0] = 2
        self.mask_path = self.tmp / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(self.mask_path))

        empty = np.zeros((nx, ny, nz), dtype=np.uint8)
        self.empty_mask_path = self.tmp / "empty_mask.nii.gz"
        nib.save(nib.Nifti1Image(empty, affine), str(self.empty_mask_path))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, **extra):
        from tools.generate_qa_snapshot import generate_qa_snapshot

        args = {"input_nifti": str(self.anat_path)}
        args.update(extra)
        # Keep every PNG inside the temp dir: generate_qa_snapshot resolves a
        # relative output_png against the process cwd, not the artifacts dir.
        png = args.get("output_png")
        if png and not Path(png).is_absolute():
            args["output_png"] = str(self.tmp / png)
        return generate_qa_snapshot(args, self.ctx)

    def test_mask_crops_the_frame_and_reports_it(self) -> None:
        res = self._run(mask_nifti=str(self.mask_path), output_png="cropped.png")
        self.assertTrue(res["ok"], res.get("error"))
        crop = res["data"]["crop"]
        self.assertTrue(crop["applied"])
        self.assertEqual(crop["shape_before"], [208, 162])
        self.assertLess(crop["shape_after"][0], 208)
        self.assertLess(crop["shape_after"][1], 162)
        self.assertGreater(
            crop["labelled_fraction_after"], 5.0 * crop["labelled_fraction_before"]
        )
        self.assertTrue(Path(res["data"]["output_png"]).exists())

    def test_no_mask_keeps_the_full_field_of_view(self) -> None:
        res = self._run(output_png="full.png")
        self.assertTrue(res["ok"], res.get("error"))
        crop = res["data"]["crop"]
        self.assertFalse(crop["applied"])
        self.assertEqual(crop["shape_before"], crop["shape_after"])
        self.assertEqual(crop["reason"], "no mask slice available")

    def test_crop_can_be_switched_off(self) -> None:
        res = self._run(
            mask_nifti=str(self.mask_path), crop_to_mask=False, output_png="off.png"
        )
        self.assertTrue(res["ok"], res.get("error"))
        crop = res["data"]["crop"]
        self.assertFalse(crop["applied"])
        self.assertEqual(crop["reason"], "crop_to_mask=false")
        self.assertEqual(crop["shape_after"], [208, 162])

    def test_empty_mask_falls_back_to_full_field_of_view(self) -> None:
        res = self._run(mask_nifti=str(self.empty_mask_path), output_png="empty.png")
        self.assertTrue(res["ok"], res.get("error"))
        crop = res["data"]["crop"]
        self.assertFalse(crop["applied"])
        self.assertIn("empty", crop["reason"])
        self.assertEqual(crop["shape_after"], [208, 162])
        self.assertTrue(Path(res["data"]["output_png"]).exists())

    def test_mismatched_mask_grid_falls_back_instead_of_crashing(self) -> None:
        import nibabel as nib

        odd = np.ones((64, 64, 4), dtype=np.uint8)
        odd_path = self.tmp / "odd_mask.nii.gz"
        nib.save(nib.Nifti1Image(odd, np.eye(4)), str(odd_path))
        res = self._run(mask_nifti=str(odd_path), output_png="odd.png")
        self.assertTrue(res["ok"], res.get("error"))
        crop = res["data"]["crop"]
        self.assertFalse(crop["applied"])
        self.assertIn("does not match", crop["reason"])

    def test_negative_margin_is_refused(self) -> None:
        res = self._run(mask_nifti=str(self.mask_path), crop_margin_frac=-0.1)
        self.assertFalse(res["ok"])
        self.assertIn("crop_margin_frac", res["error"])


if __name__ == "__main__":
    unittest.main()
