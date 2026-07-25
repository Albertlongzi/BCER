"""Readout-oversampling crop in reconstruct_grappa.

The defect these cover: CMRxRecon cine k-space is acquired with 2x readout
oversampling, and the HDF5 ``encoding_size`` / ``recon_size`` attributes report
the *uncropped* width, so nothing in the file flags it.  The crop therefore has
to be requested explicitly and reported back.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from commands.schemas import ToolContext
from tools.reconstruct_grappa import (
    _OUTER_FOV_ENERGY_HINT,
    _crop_readout,
    _outer_fov_energy_fraction,
    _parse_spacing,
    _resolve_readout_crop_target,
    reconstruct_grappa,
)

_HAS_H5PY = importlib.util.find_spec("h5py") is not None
_HAS_WRITER = (
    importlib.util.find_spec("SimpleITK") is not None
    or importlib.util.find_spec("nibabel") is not None
)

# The real P017 acquisition, from the vendor's cine_sax_info.csv:
#   FOVx=400 mm, ReconMatrix_X=208, ReadOutOversample=2
#   -> 400 / 208 = 1.923077 mm readout spacing, 418 samples stored in the H5.
P017_READOUT_SAMPLES = 418
P017_READOUT_SPACING_MM = 1.923077
P017_READOUT_FOV_MM = 400.0


class _FakeAttrs(dict):
    pass


class _FakeH5:
    def __init__(self, attrs=None):
        self.attrs = _FakeAttrs(attrs or {})


class ResolveReadoutCropTargetTests(unittest.TestCase):
    def test_no_request_means_no_crop(self) -> None:
        target, meta = _resolve_readout_crop_target(
            samples=418,
            spacing_mm=1.923077,
            spacing_source="argument",
            fov_mm=None,
            oversampling=None,
            crop_samples=None,
        )
        self.assertIsNone(target)
        self.assertEqual(meta, {"mode": "none"})

    def test_fov_mm_resolves_to_vendor_recon_matrix(self) -> None:
        target, meta = _resolve_readout_crop_target(
            samples=P017_READOUT_SAMPLES,
            spacing_mm=P017_READOUT_SPACING_MM,
            spacing_source="argument",
            fov_mm=P017_READOUT_FOV_MM,
            oversampling=None,
            crop_samples=None,
        )
        self.assertEqual(target, 208)
        self.assertEqual(meta["mode"], "fov_mm")
        self.assertAlmostEqual(meta["requested_fov_mm"], 400.0)

    def test_oversampling_factor_halves_the_readout(self) -> None:
        target, meta = _resolve_readout_crop_target(
            samples=P017_READOUT_SAMPLES,
            spacing_mm=P017_READOUT_SPACING_MM,
            spacing_source="argument",
            fov_mm=None,
            oversampling=2.0,
            crop_samples=None,
        )
        self.assertEqual(target, 209)
        self.assertEqual(meta["mode"], "oversampling_factor")

    def test_explicit_sample_count(self) -> None:
        target, meta = _resolve_readout_crop_target(
            samples=418,
            spacing_mm=1.0,
            spacing_source="default",
            fov_mm=None,
            oversampling=None,
            crop_samples=208,
        )
        self.assertEqual(target, 208)
        self.assertEqual(meta["mode"], "target_samples")

    def test_fov_mm_refuses_placeholder_spacing(self) -> None:
        # A 1.0 mm fallback would silently turn "400 mm" into "400 samples".
        with self.assertRaises(ValueError) as ctx:
            _resolve_readout_crop_target(
                samples=418,
                spacing_mm=1.0,
                spacing_source="default",
                fov_mm=400.0,
                oversampling=None,
                crop_samples=None,
            )
        self.assertIn("known readout spacing", str(ctx.exception))

    def test_over_specified_request_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _resolve_readout_crop_target(
                samples=418,
                spacing_mm=1.923077,
                spacing_source="argument",
                fov_mm=400.0,
                oversampling=2.0,
                crop_samples=None,
            )
        self.assertIn("over-specified", str(ctx.exception))

    def test_target_wider_than_axis_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_readout_crop_target(
                samples=418,
                spacing_mm=1.0,
                spacing_source="argument",
                fov_mm=None,
                oversampling=None,
                crop_samples=600,
            )

    def test_oversampling_must_exceed_one(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_readout_crop_target(
                samples=418,
                spacing_mm=1.0,
                spacing_source="argument",
                fov_mm=None,
                oversampling=1.0,
                crop_samples=None,
            )


class CropReadoutTests(unittest.TestCase):
    def test_crop_is_symmetric_and_keeps_the_fft_centre(self) -> None:
        vol = np.zeros((418, 162, 16, 12), dtype=np.float32)
        # Mark the FFT centre that _ifft2_rss produces (index n // 2).
        vol[418 // 2, 162 // 2, 3, 4] = 1.0
        cropped, start, stop = _crop_readout(vol, axis=0, target=208, np=np)
        self.assertEqual(tuple(cropped.shape), (208, 162, 16, 12))
        self.assertEqual((start, stop), (105, 313))
        self.assertEqual(418 - stop, start)  # symmetric about the centre
        self.assertEqual(cropped[208 // 2, 162 // 2, 3, 4], 1.0)

    def test_crop_on_a_non_zero_axis(self) -> None:
        vol = np.zeros((162, 418, 16), dtype=np.float32)
        vol[80, 418 // 2, 2] = 1.0
        cropped, start, _ = _crop_readout(vol, axis=1, target=208, np=np)
        self.assertEqual(tuple(cropped.shape), (162, 208, 16))
        self.assertEqual(cropped[80, 208 // 2, 2], 1.0)
        self.assertEqual(start, 105)


class OuterFovEnergyTests(unittest.TestCase):
    def test_centre_concentrated_energy_reads_low(self) -> None:
        vol = np.zeros((418, 162), dtype=np.float32)
        vol[200:220, 70:90] = 1.0
        frac = _outer_fov_energy_fraction(vol, axis=0, np=np)
        self.assertIsNotNone(frac)
        self.assertLess(frac, 0.01)

    def test_uniform_energy_reads_high(self) -> None:
        vol = np.ones((418, 162), dtype=np.float32)
        frac = _outer_fov_energy_fraction(vol, axis=0, np=np)
        self.assertGreater(frac, 0.4)
        self.assertGreater(frac, _OUTER_FOV_ENERGY_HINT)

    def test_hint_threshold_separates_the_measured_p017_cases(self) -> None:
        # Measured on Center006_Siemens_30T_Prisma_P017_cine_sax:
        #   uncropped readout 0.0231 / cropped readout 0.2835 / phase axis 0.4567
        self.assertLess(0.0231, _OUTER_FOV_ENERGY_HINT)
        self.assertGreater(0.2835, _OUTER_FOV_ENERGY_HINT)
        self.assertGreater(0.4567, _OUTER_FOV_ENERGY_HINT)

    def test_empty_volume_returns_none(self) -> None:
        self.assertIsNone(_outer_fov_energy_fraction(np.zeros((418, 162)), axis=0, np=np))


class ParseSpacingSourceTests(unittest.TestCase):
    def test_argument_source(self) -> None:
        spacing, source = _parse_spacing(_FakeH5(), [1.923077, 2.492877, 10.0])
        self.assertEqual(source, "argument")
        self.assertAlmostEqual(spacing[0], 1.923077)

    def test_h5_attr_source(self) -> None:
        spacing, source = _parse_spacing(_FakeH5({"pixel_spacing": [2.0, 3.0, 4.0]}), None)
        self.assertEqual(source, "h5_attr:pixel_spacing")
        self.assertEqual(spacing, [2.0, 3.0, 4.0])

    def test_default_source_is_flagged(self) -> None:
        spacing, source = _parse_spacing(_FakeH5(), None)
        self.assertEqual(source, "default")
        self.assertEqual(spacing, [1.0, 1.0, 1.0])


@unittest.skipUnless(_HAS_H5PY and _HAS_WRITER, "h5py and a NIfTI writer are required")
class ReadoutCropEndToEndTests(unittest.TestCase):
    """Drives the whole tool on a miniature H5 that mimics the CMRxRecon layout."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="bcer_readout_crop_"))
        self.ctx = ToolContext(
            case_id="c",
            run_id="r",
            run_dir=self.tmp,
            artifacts_dir=self.tmp / "artifacts",
            case_state_path=self.tmp / "case_state.json",
        )
        self.ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)

        import h5py

        # (frames, slices, readout, phase) real image volume, 2x oversampled
        # readout: the signal only occupies the central half.
        self.h5_path = self.tmp / "mini_cine_sax.h5"
        vol = np.zeros((3, 4, 64, 40), dtype=np.float32)
        vol[:, :, 24:40, 14:26] = 1.0
        with h5py.File(str(self.h5_path), "w") as f:
            f.create_dataset("reconstruction_rss", data=vol)
            # Both attrs report the UNCROPPED width, exactly as the real files do.
            f.attrs["encoding_size"] = np.array([64, 40, 1])
            f.attrs["recon_size"] = np.array([64, 40, 1])

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_back(self, path: str):
        try:
            import SimpleITK as sitk  # type: ignore

            img = sitk.ReadImage(path)
            return list(img.GetSize()), list(img.GetSpacing()), list(img.GetOrigin())
        except ImportError:
            import nibabel as nib  # type: ignore

            img = nib.load(path)
            aff = img.affine
            return (
                list(img.shape),
                [float(aff[i, i]) for i in range(3)],
                [float(aff[i, 3]) for i in range(3)],
            )

    def test_no_crop_by_default_and_origin_stays_at_zero(self) -> None:
        res = reconstruct_grappa(
            {
                "h5_path": str(self.h5_path),
                "image_key": "reconstruction_rss",
                "pixel_spacing": [2.0, 3.0, 10.0],
            },
            self.ctx,
        )
        self.assertTrue(res["ok"], res.get("error"))
        data = res["data"]
        self.assertEqual(data["mode"], "image_passthrough")
        # _normalize_image_dataset_to_vol puts the H5's last axis first.
        self.assertEqual(data["output_shape"], [40, 64, 4, 3])
        self.assertFalse(data["readout_crop"]["applied"])
        self.assertEqual(data["readout_crop"]["mode"], "none")
        size, _, origin = self._read_back(data["reconstructed_nifti"])
        self.assertEqual(size[:3], [40, 64, 4])
        self.assertEqual([round(o, 6) for o in origin[:3]], [0.0, 0.0, 0.0])

    def test_uncropped_oversampling_is_reported_as_a_warning_not_acted_on(self) -> None:
        res = reconstruct_grappa(
            {
                "h5_path": str(self.h5_path),
                "image_key": "reconstruction_rss",
                "readout_axis": 1,
                "pixel_spacing": [3.0, 2.0, 10.0],
            },
            self.ctx,
        )
        self.assertTrue(res["ok"], res.get("error"))
        self.assertFalse(res["data"]["readout_crop"]["applied"])
        self.assertLess(
            res["data"]["readout_crop"]["outer_half_energy_fraction"],
            _OUTER_FOV_ENERGY_HINT,
        )
        self.assertTrue(
            any("readout_fov_mm" in w for w in res["warnings"]),
            res["warnings"],
        )

    def test_explicit_fov_crop_shifts_the_origin(self) -> None:
        res = reconstruct_grappa(
            {
                "h5_path": str(self.h5_path),
                "image_key": "reconstruction_rss",
                "readout_axis": 1,
                "readout_fov_mm": 64.0,  # 64 mm / 2 mm = 32 samples of 64
                "pixel_spacing": [3.0, 2.0, 10.0],
            },
            self.ctx,
        )
        self.assertTrue(res["ok"], res.get("error"))
        crop = res["data"]["readout_crop"]
        self.assertTrue(crop["applied"])
        self.assertEqual(crop["mode"], "fov_mm")
        self.assertEqual(crop["axis"], 1)
        self.assertEqual(crop["samples_before"], 64)
        self.assertEqual(crop["samples_after"], 32)
        self.assertEqual(crop["removed_left"], 16)
        self.assertEqual(crop["removed_right"], 16)
        self.assertAlmostEqual(crop["fov_before_mm"], 128.0)
        self.assertAlmostEqual(crop["fov_after_mm"], 64.0)
        self.assertAlmostEqual(crop["origin_shift_mm"], 32.0)
        self.assertEqual(res["data"]["output_shape"], [40, 32, 4, 3])

        size, spacing, origin = self._read_back(res["data"]["reconstructed_nifti"])
        self.assertEqual(size[:3], [40, 32, 4])
        self.assertAlmostEqual(spacing[1], 2.0, places=5)
        # The retained block keeps its world position: 16 discarded samples x 2 mm.
        self.assertAlmostEqual(origin[1], 32.0, places=4)
        self.assertAlmostEqual(origin[0], 0.0, places=6)

    def test_fov_crop_without_spacing_is_refused(self) -> None:
        res = reconstruct_grappa(
            {
                "h5_path": str(self.h5_path),
                "image_key": "reconstruction_rss",
                "readout_axis": 1,
                "readout_fov_mm": 64.0,
            },
            self.ctx,
        )
        self.assertFalse(res["ok"])
        self.assertIn("known readout spacing", res["error"])

    def test_over_specified_crop_is_refused(self) -> None:
        res = reconstruct_grappa(
            {
                "h5_path": str(self.h5_path),
                "image_key": "reconstruction_rss",
                "readout_oversampling": 2.0,
                "readout_crop_samples": 32,
            },
            self.ctx,
        )
        self.assertFalse(res["ok"])
        self.assertIn("over-specified", res["error"])

    @unittest.skipUnless(
        importlib.util.find_spec("pygrappa") is not None, "pygrappa is required"
    )
    def test_grappa_mode_crops_axis_zero_and_keeps_undersampling(self) -> None:
        import h5py

        ks_path = self.tmp / "mini_kspace.h5"
        rng = np.random.RandomState(7)
        # (frames, slices, coils, kx, ky) -- the CMRxRecon cine layout.
        ks = (rng.rand(2, 2, 4, 32, 16) + 1j * rng.rand(2, 2, 4, 32, 16)).astype(np.complex64)
        with h5py.File(str(ks_path), "w") as f:
            f.create_dataset("kspace", data=ks)

        res = reconstruct_grappa(
            {
                "h5_path": str(ks_path),
                "coil_axis": 2,
                "nonspatial_order": [1, 0],
                "acs_lines": 6,
                "kernel_size": [3, 3],
                "undersample_factor": 2,
                "readout_crop_samples": 16,
                "pixel_spacing": [2.0, 3.0, 10.0],
            },
            self.ctx,
        )
        self.assertTrue(res["ok"], res.get("error"))
        data = res["data"]
        self.assertEqual(data["mode"], "grappa")
        # GRAPPA mode builds (kx, ky, slices, frames), so the readout is axis 0.
        self.assertEqual(data["output_shape"], [16, 16, 2, 2])
        crop = data["readout_crop"]
        self.assertTrue(crop["applied"])
        self.assertEqual(crop["axis"], 0)
        self.assertEqual((crop["samples_before"], crop["samples_after"]), (32, 16))
        self.assertAlmostEqual(crop["origin_shift_mm"], 16.0)
        # The retrospective undersampling block is untouched by the crop.
        self.assertTrue(data["undersample"]["applied"])
        self.assertEqual(data["undersample"]["factor"], 2)
        self.assertEqual(data["undersample"]["ky_lines_total"], 16)
        # The zero-filled reference is cropped identically.
        zf_size, _, zf_origin = self._read_back(data["zerofilled_nifti"])
        self.assertEqual(zf_size[:3], [16, 16, 2])
        self.assertAlmostEqual(zf_origin[0], 16.0, places=4)

    def test_readout_axis_out_of_range_is_refused(self) -> None:
        res = reconstruct_grappa(
            {
                "h5_path": str(self.h5_path),
                "image_key": "reconstruction_rss",
                "readout_axis": 9,
                "readout_crop_samples": 32,
            },
            self.ctx,
        )
        self.assertFalse(res["ok"])
        self.assertIn("out of range", res["error"])


if __name__ == "__main__":
    unittest.main()
