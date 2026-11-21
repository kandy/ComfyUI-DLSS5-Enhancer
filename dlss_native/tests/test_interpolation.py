"""Tests for optical-flow frame interpolation (needs the GPU).

Run from the workspace root::

    python3 dlss_native/tests/test_interpolation.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import dlss_native as D

MAX_FRAMES = 3


def _texture(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A smooth random RGB texture with unique local structure for flow."""
    rng = np.random.default_rng(seed)
    noise = rng.random((height, width, 3)).astype(np.float32)
    # A light blur keeps the texture unique locally without making it ambiguous.
    noise = (
        np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
        + np.roll(noise, 1, 1) + np.roll(noise, -1, 1) + noise
    ) / 5.0
    noise = (noise - noise.min()) / max(1e-6, noise.max() - noise.min())
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(noise * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def _shift(rgba: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate content by (dx, dy) with edge replication (no wraparound)."""
    height, width = rgba.shape[:2]
    ys = np.clip(np.arange(height) - dy, 0, height - 1)
    xs = np.clip(np.arange(width) - dx, 0, width - 1)
    return np.ascontiguousarray(rgba[ys][:, xs])


def _center_rmse(a: np.ndarray, b: np.ndarray, border: int = 24) -> float:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) != (hb, wb):
        raise ValueError("shapes differ")
    a = a[border:ha - border, border:wa - border, :3].astype(np.float64)
    b = b[border:hb - border, border:wb - border, :3].astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


@unittest.skipUnless(D.interp_available(), "NVIDIA Optical Flow Accelerator unavailable")
class InterpolationTest(unittest.TestCase):
    def test_mid_frame_matches_true_motion(self) -> None:
        prev = _texture(512, 512)
        nxt = _shift(prev, dx=8, dy=2)
        with D.NativeDLSS() as dlss:
            mids = dlss.interpolate(prev, nxt, count=1, refine=False)
            self.assertEqual(dlss.interp_last_path(), "optical_flow")
        self.assertEqual(len(mids), 1)
        self.assertEqual(mids[0].shape, prev.shape)
        expected = _shift(prev, dx=4, dy=1)
        rmse = _center_rmse(mids[0], expected)
        reference = _center_rmse(prev, expected)
        print(f"\n[mid-frame] RMSE={rmse:.2f} (uninterpolated={reference:.2f})")
        self.assertLess(rmse, 4.0)
        self.assertLess(rmse, reference * 0.25)

    def test_second_order_accelerating_motion(self) -> None:
        base = _texture(512, 512, seed=4)
        # Positions at t=-1, 0, 1 are 0, 8, 32: velocity 16 and acceleration
        # 16 pixels per frame.  The true midpoint is 18, while pairwise motion
        # assumes the interval average (20).
        prev2 = _shift(base, dx=0, dy=0)
        prev = _shift(base, dx=8, dy=0)
        nxt = _shift(base, dx=32, dy=0)
        expected = _shift(base, dx=18, dy=0)

        with D.NativeDLSS() as dlss:
            pairwise = dlss.interpolate(prev, nxt, count=1, refine=False)[0]
            self.assertEqual(dlss.interp_last_order(), 1)

            second_order = dlss.interpolate(
                prev, nxt, count=1, refine=False, prev2=prev2
            )[0]
            self.assertEqual(dlss.interp_last_order(), 2)
            self.assertEqual(dlss.interp_last_path(), "optical_flow")

        pairwise_rmse = _center_rmse(pairwise, expected)
        second_order_rmse = _center_rmse(second_order, expected)
        print(
            f"\n[second-order] pairwise RMSE={pairwise_rmse:.2f} "
            f"three-frame RMSE={second_order_rmse:.2f}"
        )
        self.assertLess(second_order_rmse, 4.0)
        self.assertLess(second_order_rmse, pairwise_rmse * 0.5)

    def test_all_counts_and_refine(self) -> None:
        prev = _texture(512, 512, seed=1)
        nxt = _shift(prev, dx=-5, dy=3)
        with D.NativeDLSS() as dlss:
            for count in range(1, MAX_FRAMES + 1):
                for refine in (False, True):
                    mids = dlss.interpolate(prev, nxt, count=count, refine=refine)
                    self.assertEqual(len(mids), count)
                    for frame in mids:
                        self.assertEqual(frame.shape, prev.shape)
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertGreater(float(frame[..., :3].std()), 1.0)
                    self.assertEqual(dlss.interp_last_path(), "optical_flow")

    def test_count_bounds(self) -> None:
        prev = _texture(512, 512, seed=2)
        with D.NativeDLSS() as dlss:
            for bad in (0, MAX_FRAMES + 1):
                with self.assertRaises(ValueError):
                    dlss.interpolate(prev, prev, count=bad)
            with self.assertRaises(ValueError):
                dlss.interpolate(prev, _texture(513, 512), count=1)
            with self.assertRaises(ValueError):
                dlss.interpolate(
                    prev, prev, count=1, prev2=_texture(513, 512)
                )

    def test_odd_dimensions(self) -> None:
        prev = _texture(611, 517, seed=3)
        nxt = _shift(prev, dx=8, dy=2)
        with D.NativeDLSS() as dlss:
            mids = dlss.interpolate(prev, nxt, count=1, refine=False)
            self.assertEqual(dlss.interp_last_path(), "optical_flow")
        expected = _shift(prev, dx=4, dy=1)
        rmse = _center_rmse(mids[0], expected, border=32)
        reference = _center_rmse(prev, expected, border=32)
        print(f"\n[odd dims] RMSE={rmse:.2f} (uninterpolated={reference:.2f})")
        self.assertLess(rmse, 12.0)
        self.assertLess(rmse, reference * 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
