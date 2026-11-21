"""Unit tests for the MFG helpers that need no GPU or ComfyUI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from dlss5.interpolation import FrameInterpolator  # noqa: E402
from dlss5.multiframe import (  # noqa: E402
    MFG_FACTORS,
    MultiFrameGenerator,
    frames_after_mfg,
    generated_per_pair,
    resolve_mfg_factor,
)


class FactorTest(unittest.TestCase):
    def test_labels(self) -> None:
        for label, factor in MFG_FACTORS.items():
            self.assertEqual(resolve_mfg_factor(label), factor)
        self.assertEqual(resolve_mfg_factor(3), 3)

    def test_rejects_unknown(self) -> None:
        for bad in ("1x", "5x", "off", 1, 5, 0):
            with self.assertRaises(ValueError):
                resolve_mfg_factor(bad)


class CountTest(unittest.TestCase):
    def test_generated_per_pair(self) -> None:
        self.assertEqual(generated_per_pair("2x"), 1)
        self.assertEqual(generated_per_pair("3x"), 2)
        self.assertEqual(generated_per_pair("4x"), 3)

    def test_total_length(self) -> None:
        self.assertEqual(frames_after_mfg(1, "4x"), 1)
        self.assertEqual(frames_after_mfg(2, "2x"), 3)
        self.assertEqual(frames_after_mfg(3, "3x"), 7)
        self.assertEqual(frames_after_mfg(4, "4x"), 13)

    def test_rejects_negative_source(self) -> None:
        with self.assertRaises(ValueError):
            frames_after_mfg(-1, "2x")


class ThreeFrameHistoryTest(unittest.TestCase):
    @staticmethod
    def _frame(value: int) -> np.ndarray:
        return np.full((2, 2, 4), value, dtype=np.uint8)

    def test_passes_n_minus_two_from_second_pair(self) -> None:
        seen: list[int | None] = []

        def fake_between(self, prev, next, factor, prev2=None):
            seen.append(None if prev2 is None else int(prev2[0, 0, 0]))
            return [np.zeros_like(prev) for _ in range(int(factor) - 1)]

        frames = [self._frame(value) for value in (1, 2, 3, 4)]
        with mock.patch.object(FrameInterpolator, "between", fake_between):
            with MultiFrameGenerator(refine=False) as generator:
                for previous, current in zip(frames, frames[1:]):
                    generator.between(previous, current, "2x")

        self.assertEqual(seen, [None, 1, 2])

    def test_scene_cut_clears_history(self) -> None:
        seen: list[int | None] = []

        def fake_between(self, prev, next, factor, prev2=None):
            seen.append(None if prev2 is None else int(prev2[0, 0, 0]))
            return [np.zeros_like(prev) for _ in range(int(factor) - 1)]

        frames = [self._frame(value) for value in (1, 2, 3, 4)]
        with mock.patch.object(FrameInterpolator, "between", fake_between):
            with MultiFrameGenerator(refine=False) as generator:
                generator.between(frames[0], frames[1], "2x", scene_cut=True)
                generator.between(frames[1], frames[2], "2x")
                generator.between(frames[2], frames[3], "2x")

        self.assertEqual(seen, [None, 2])


class SceneCutTest(unittest.TestCase):
    def test_holds_previous_frame_instead_of_blending(self) -> None:
        previous = np.zeros((2, 2, 4), dtype=np.uint8)
        current = np.full((2, 2, 4), 255, dtype=np.uint8)
        with MultiFrameGenerator(refine=False) as generator:
            mids = generator.between(previous, current, "4x", scene_cut=True)
            self.assertEqual(len(mids), 3)
            for mid in mids:
                self.assertTrue(np.array_equal(mid, previous))
            self.assertEqual(generator.held_frames, 3)
            self.assertEqual(generator.generated_frames, 0)
            self.assertEqual(generator.path, "scene_cut")

    def test_rejects_mismatched_shapes(self) -> None:
        generator = MultiFrameGenerator(refine=False)
        try:
            with self.assertRaises(ValueError):
                generator.between(
                    np.zeros((2, 2, 4), dtype=np.uint8),
                    np.zeros((3, 3, 4), dtype=np.uint8),
                    "2x",
                )
        finally:
            generator.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
