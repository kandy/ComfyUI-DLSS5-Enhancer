"""Regression tests for the native in-process DLSS binding.

Run from the workspace root (needs the GPU)::

    python3 dlss_native/tests/test_native_dlss.py

Covers: binding availability, the temporal ``upscale_ex`` entrypoint
(motion + reset + quality + preset plumbing), neural-vs-fallback
reporting, and a golden comparison against the Windows DLSS 2x output.
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

import cv2

from dlss_native import NativeDLSS, is_available

# Test images: override with DLSS_TESTDATA_DIR (the plugin's bundled copy
# has no test_output/ next to it).
_TESTDATA = Path(
    os.environ.get(
        "DLSS_TESTDATA_DIR",
        WORKSPACE / "test_output",
    )
)
GOLDEN = _TESTDATA / "test_upscaled_2.jpeg"
INPUT = _TESTDATA / "test.jpeg"


def _load_rgba(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"cannot load {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


class NativeDlssTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert is_available(), "native DLSS binding unavailable"
        if not INPUT.is_file():
            raise unittest.SkipTest(f"test data missing in {_TESTDATA}")
        img = _load_rgba(INPUT)
        # Keep the test fast: centre crop; the golden comparison below
        # uses the full frame instead.
        h, w = img.shape[:2]
        ch, cw = 512, 512
        cls.crop = np.ascontiguousarray(
            img[h // 2 - ch // 2 : h // 2 + ch // 2,
                w // 2 - cw // 2 : w // 2 + cw // 2]
        )

    def test_temporal_entrypoint_is_neural(self) -> None:
        h, w = self.crop.shape[:2]
        mv = np.zeros((h, w, 2), dtype=np.float16)
        with NativeDLSS() as dlss:
            out = dlss.upscale_ex(
                self.crop, factor=2.0, motion=mv, reset=True,
                quality=0, preset=13,
            )
            self.assertEqual(out.shape, (h * 2, w * 2, 4))
            self.assertEqual(dlss.last_path(), "neural")
            out2 = dlss.upscale_ex(
                self.crop, factor=2.0, motion=mv, reset=False,
                quality=0, preset=13,
            )
            self.assertEqual(dlss.last_path(), "neural")
        # Sanity: output is an image, not blank.
        self.assertGreater(float(out2[..., :3].std()), 1.0)

    def test_golden_full_frame(self) -> None:
        if not GOLDEN.is_file() or not INPUT.is_file():
            self.skipTest(f"test data missing in {_TESTDATA}")
        img = _load_rgba(INPUT)
        gold = cv2.imread(str(GOLDEN), cv2.IMREAD_COLOR)
        self.assertIsNotNone(gold, f"cannot load {GOLDEN}")
        with NativeDLSS() as dlss:
            out = dlss.upscale(img, factor=2.0)
            self.assertEqual(dlss.last_path(), "neural")
        bgr = cv2.cvtColor(out, cv2.COLOR_RGBA2BGR)
        if gold.shape[:2] != bgr.shape[:2]:
            gold = cv2.resize(gold, (bgr.shape[1], bgr.shape[0]))
        diff = bgr.astype(np.float64) - gold.astype(np.float64)
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        print(f"\n[golden] RMSE={rmse:.2f}")
        # Generous: JPEG recompression + single Reset=1 frame + preset
        # differences account for several RMSE points (reference: ~8.0).
        self.assertLess(rmse, 15.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
