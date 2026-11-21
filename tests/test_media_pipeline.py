"""End-to-end media test for the video interpolation pipeline.

Needs PyAV (``pip install av``), ffmpeg/ffprobe and a working Optical Flow
Accelerator; skips cleanly otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

try:
    import av  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    av = None

from dlss5.interpolation import FrameInterpolator, interpolated_timeline  # noqa: E402
from dlss5.media import RawFrameEncoder, decode_frames, probe_video  # noqa: E402
from dlss5.paths import RuntimeLayout  # noqa: E402


def _native_ready() -> bool:
    try:
        from dlss5 import native

        return bool(native.load().interp_available())
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(av is not None, "PyAV is not installed")
@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg missing")
@unittest.skipUnless(_native_ready(), "Optical Flow Accelerator unavailable")
class VideoPipelineTest(unittest.TestCase):
    def test_interpolated_encode_matches_source_timing(self) -> None:
        factor, limit = 3, 6
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.mp4"
            target = Path(tmp) / "out.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=640x360:rate=30",
                    "-t",
                    "1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )

            layout = RuntimeLayout(root=Path("."), config={})
            meta = probe_video(layout, source)
            self.assertEqual(int(meta["frames"]), 30)

            encoder = RawFrameEncoder(
                layout,
                target,
                codec="H.264",
                quality="Good",
                width=meta["width"],
                height=meta["height"],
                rate=meta["rate"] * factor,
                time_base=meta["time_base"] / factor,
            )
            interpolator = FrameInterpolator(refine=True)
            written = 0
            previous = None
            previous_pts = 0
            try:
                for _index, rgba, pts in decode_frames(
                    source, meta["rotation"], limit=limit
                ):
                    if previous is None:
                        timeline = [int(pts) * factor]
                    else:
                        mids = interpolator.between(previous, rgba, factor)
                        timeline = interpolated_timeline(previous_pts, pts, factor)
                        for mid, mid_pts in zip(mids, timeline[:-1]):
                            encoder.write(mid, mid_pts)
                            written += 1
                    encoder.write(rgba, timeline[-1])
                    written += 1
                    previous, previous_pts = rgba, pts
            finally:
                interpolator.close()
                encoder.close()

            out = probe_video(layout, target)
            self.assertEqual(written, 1 + (limit - 1) * factor)
            self.assertEqual(int(out["frames"]), written)
            self.assertEqual(int(out["width"]), int(meta["width"]))
            self.assertEqual(int(out["height"]), int(meta["height"]))
            self.assertEqual(out["rate"], meta["rate"] * factor)
            source_span = float(limit * meta["time_base"])
            out_span = float(out["frames"] * out["time_base"])
            # (N - 1) * F + 1 frames: short by less than one source frame.
            tolerance = 1.1 / float(meta["rate"])
            self.assertLess(abs(out_span - source_span), tolerance)
            self.assertEqual(interpolator.path, "optical_flow")


if __name__ == "__main__":
    unittest.main(verbosity=2)
