from __future__ import annotations

import unittest

import numpy as np

from tools.reconstruct_grappa import _extract_calib_frame, _infer_axes, _normalize_image_dataset_to_vol


class ReconstructGrappaCalibTests(unittest.TestCase):
    def test_infer_axes_rejects_invalid_or_colliding_coil_hint(self) -> None:
        shape = (4, 3, 5, 16, 12)
        with self.assertRaises(ValueError):
            _infer_axes(shape, coil_axis_hint=3)  # collides with inferred kx axis
        with self.assertRaises(ValueError):
            _infer_axes(shape, coil_axis_hint=4)  # collides with inferred ky axis
        with self.assertRaises(ValueError):
            _infer_axes(shape, coil_axis_hint=99)

    def test_extract_calib_frame_from_full_rank_5d(self) -> None:
        full_shape = (4, 3, 5, 16, 12)
        nonspatial, coil_ax, kx_ax, ky_ax = _infer_axes(full_shape)
        self.assertEqual((nonspatial, coil_ax, kx_ax, ky_ax), ([0, 1], 2, 3, 4))

        calib = np.zeros((4, 3, 5, 16, 4), dtype=np.complex64)
        frame = _extract_calib_frame(
            calib_ds=calib,
            idx_tuple=(2, 1),
            full_shape=full_shape,
            full_nonspatial_axes=nonspatial,
            n_coils=5,
            kx_size=16,
        )
        self.assertEqual(tuple(frame.shape), (16, 4, 5))

    def test_extract_calib_frame_from_reduced_rank_5d(self) -> None:
        full_shape = (4, 3, 5, 16, 12)
        nonspatial, _, _, _ = _infer_axes(full_shape)

        # Missing the phase axis, keeps slice + coil + kx + ky_calib.
        calib = np.zeros((4, 5, 16, 4), dtype=np.complex64)
        frame = _extract_calib_frame(
            calib_ds=calib,
            idx_tuple=(3, 1),
            full_shape=full_shape,
            full_nonspatial_axes=nonspatial,
            n_coils=5,
            kx_size=16,
        )
        self.assertEqual(tuple(frame.shape), (16, 4, 5))

    def test_extract_calib_frame_from_global_3d(self) -> None:
        full_shape = (4, 3, 5, 16, 12)
        nonspatial, _, _, _ = _infer_axes(full_shape)

        calib = np.zeros((5, 16, 4), dtype=np.complex64)  # (coil, kx, ky_calib)
        frame = _extract_calib_frame(
            calib_ds=calib,
            idx_tuple=(0, 0),
            full_shape=full_shape,
            full_nonspatial_axes=nonspatial,
            n_coils=5,
            kx_size=16,
        )
        self.assertEqual(tuple(frame.shape), (16, 4, 5))

    def test_extract_calib_frame_from_6d_reduced_rank(self) -> None:
        full_shape = (2, 3, 4, 5, 16, 12)
        nonspatial, coil_ax, kx_ax, ky_ax = _infer_axes(full_shape)
        self.assertEqual((nonspatial, coil_ax, kx_ax, ky_ax), ([0, 1, 2], 3, 4, 5))

        # Missing one nonspatial axis (phase): (slice, echo, coil, kx, ky_calib)
        calib = np.zeros((2, 4, 5, 16, 4), dtype=np.complex64)
        frame = _extract_calib_frame(
            calib_ds=calib,
            idx_tuple=(1, 2, 3),
            full_shape=full_shape,
            full_nonspatial_axes=nonspatial,
            n_coils=5,
            kx_size=16,
        )
        self.assertEqual(tuple(frame.shape), (16, 4, 5))

    def test_normalize_image_dataset_to_vol_3d(self) -> None:
        arr = np.zeros((10, 256, 216), dtype=np.float32)  # (z, y, x)
        vol = _normalize_image_dataset_to_vol(arr, np)
        self.assertEqual(tuple(vol.shape), (216, 256, 10))

    def test_normalize_image_dataset_to_vol_2d(self) -> None:
        arr = np.zeros((128, 96), dtype=np.float32)  # (y, x)
        vol = _normalize_image_dataset_to_vol(arr, np)
        self.assertEqual(tuple(vol.shape), (96, 128, 1))


if __name__ == "__main__":
    unittest.main()
