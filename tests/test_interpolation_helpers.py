"""Unit tests for the interpolation helpers that need no GPU or ComfyUI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from dlss5.interpolation import (  # noqa: E402
    INTERPOLATION_FACTORS,
    _crossfade,
    interpolated_timeline,
    resolve_factor,
)


class TimelineTest(unittest.TestCase):
    def test_constant_interval(self) -> None:
        self.assertEqual(interpolated_timeline(100, 101, 2), [201, 202])
        self.assertEqual(interpolated_timeline(100, 101, 3), [301, 302, 303])
        self.assertEqual(interpolated_timeline(100, 101, 4), [401, 402, 403, 404])

    def test_variable_interval_preserves_duration(self) -> None:
        # A source interval that is not a multiple of the factor must still end
        # exactly on the real frame.
        for factor in (2, 3, 4):
            slots = interpolated_timeline(100, 105, factor)
            self.assertEqual(len(slots), factor)
            self.assertEqual(slots[-1], 105 * factor)
            self.assertEqual(slots[0], 100 * factor + 5)
            # Monotonic and equal spacing.
            deltas = np.diff([100 * factor, *slots])
            self.assertTrue(np.all(deltas == 5))

    def test_duration_matches_source(self) -> None:
        time_base = 1.0 / 1000.0
        for factor in (1, 2, 3, 4):
            slots = interpolated_timeline(0, 1000, factor)
            out_time_base = time_base / factor
            duration = slots[-1] * out_time_base
            self.assertAlmostEqual(duration, 1.0, places=9)
            self.assertEqual(len(slots), max(1, factor))

    def test_rejects_bad_factor(self) -> None:
        with self.assertRaises(ValueError):
            interpolated_timeline(0, 1, 0)


class FactorTest(unittest.TestCase):
    def test_labels(self) -> None:
        for label, factor in INTERPOLATION_FACTORS.items():
            self.assertEqual(resolve_factor(label), factor)
        self.assertEqual(resolve_factor(3), 3)

    def test_rejects_unknown(self) -> None:
        for bad in ("5x", "off-by-one", 5, 0):
            if bad == "off-by-one":
                pass
            with self.assertRaises(ValueError):
                resolve_factor(bad)


class CrossfadeTest(unittest.TestCase):
    def test_midpoint(self) -> None:
        prev = np.zeros((4, 4, 4), dtype=np.uint8)
        nxt = np.full((4, 4, 4), 200, dtype=np.uint8)
        mid = _crossfade(prev, nxt, 0.5)
        self.assertEqual(mid.dtype, np.uint8)
        self.assertTrue(np.all(mid == 100))

    def test_endpoints(self) -> None:
        prev = np.full((2, 2, 4), 10, dtype=np.uint8)
        nxt = np.full((2, 2, 4), 250, dtype=np.uint8)
        self.assertTrue(np.all(_crossfade(prev, nxt, 0.0) == 10))
        self.assertTrue(np.all(_crossfade(prev, nxt, 1.0) == 250))


if __name__ == "__main__":
    unittest.main(verbosity=2)
