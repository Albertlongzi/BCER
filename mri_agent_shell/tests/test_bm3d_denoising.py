from __future__ import annotations

import unittest

import numpy as np

from tools.bm3d_denoising import _denormalize, _denoise_slicewise, _minmax_normalize


def _identity_bm3d(arr: np.ndarray, sigma_psd: float, profile: str = "np") -> np.ndarray:
    del sigma_psd, profile
    return np.asarray(arr, dtype=np.float32)


class BM3DDenoisingTests(unittest.TestCase):
    def test_minmax_normalize_unit_range(self) -> None:
        arr = np.array([[[0.0, 2.0], [4.0, 6.0]]], dtype=np.float32)
        norm, vmin, vmax = _minmax_normalize(arr)
        self.assertEqual(float(vmin), 0.0)
        self.assertEqual(float(vmax), 6.0)
        self.assertGreaterEqual(float(np.min(norm)), 0.0)
        self.assertLessEqual(float(np.max(norm)), 1.0)

    def test_minmax_normalize_constant(self) -> None:
        arr = np.full((4, 4, 3), 7.5, dtype=np.float32)
        norm, vmin, vmax = _minmax_normalize(arr)
        self.assertTrue(np.allclose(norm, 0.0))
        self.assertEqual(float(vmin), 7.5)
        self.assertEqual(float(vmax), 7.5)

    def test_denormalize_round_trip(self) -> None:
        src = np.linspace(0.0, 10.0, num=24, dtype=np.float32).reshape((2, 3, 4))
        norm, vmin, vmax = _minmax_normalize(src)
        rec = _denormalize(norm, vmin, vmax)
        self.assertTrue(np.allclose(src, rec, atol=1e-5))

    def test_denoise_slicewise_calls_2d_profile(self) -> None:
        arr = np.random.rand(3, 10, 8).astype(np.float32)
        out = _denoise_slicewise(arr, 0.08, _identity_bm3d)
        self.assertEqual(tuple(out.shape), tuple(arr.shape))
        self.assertTrue(np.allclose(out, arr, atol=1e-6))

    def test_denoise_slicewise_2d_input(self) -> None:
        arr = np.random.rand(12, 9).astype(np.float32)
        out = _denoise_slicewise(arr, 0.08, _identity_bm3d)
        self.assertEqual(tuple(out.shape), tuple(arr.shape))
        self.assertTrue(np.allclose(out, arr, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
