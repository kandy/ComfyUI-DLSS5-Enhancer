"""Multi Frame Generation over an IMAGE batch.

DLSS Multi Frame Generation is a swapchain feature: the game renders one frame
and DLSS-G synthesises up to three more before the next rendered frame. The
CUDA SDK exposes no such entry point outside a swapchain (Linux reports
``FrameGeneration.Available = 0``), so this module implements the same contract
from the frames the pack can see. The NVIDIA Optical Flow Accelerator warps
both neighbours and the DLSS neural engine refines every synthesised frame.
"""

from __future__ import annotations

import numpy as np

from .interpolation import FrameInterpolator, MAX_INTERPOLATION_FACTOR

MFG_FACTORS: dict[str, int] = {"2x": 2, "3x": 3, "4x": 4}
MFG_LABELS: list[str] = list(MFG_FACTORS)
MAX_MFG_FACTOR = MAX_INTERPOLATION_FACTOR


def resolve_mfg_factor(label_or_factor: str | int) -> int:
    """Translate the node's combo label (or a raw multiplier) to a factor."""
    if isinstance(label_or_factor, str):
        try:
            return MFG_FACTORS[label_or_factor]
        except KeyError as exc:
            choices = ", ".join(MFG_LABELS)
            raise ValueError(
                f"Unknown multi frame generation mode {label_or_factor!r}. "
                f"Choose one of: {choices}."
            ) from exc
    value = int(label_or_factor)
    if value not in MFG_FACTORS.values():
        choices = ", ".join(MFG_LABELS)
        raise ValueError(
            f"Unsupported multi frame generation factor {value}. Choose one of: {choices}."
        )
    return value


def generated_per_pair(factor: str | int) -> int:
    """How many synthesised frames MFG inserts between two source frames."""
    return resolve_mfg_factor(factor) - 1


def frames_after_mfg(source_frames: int, factor: str | int) -> int:
    """Total output length for ``source_frames`` inputs at ``factor``.

    The usual convention applies: ``(N - 1) * factor + 1`` frames, so the
    result still ends on the last real frame and MFG never invents a frame
    after it.
    """
    source_frames = int(source_frames)
    if source_frames < 0:
        raise ValueError("source_frames must be >= 0")
    if source_frames <= 1:
        return source_frames
    return (source_frames - 1) * resolve_mfg_factor(factor) + 1


class MultiFrameGenerator:
    """Insert MFG frames between consecutive RGBA8 frames.

    Generates the same ``factor - 1`` frames as the interpolation helper, but
    keeps MFG-specific accounting and holds the outgoing frame across a scene
    cut instead of blending two unrelated shots into a ghost.

    Calls are expected to arrive in playback order. The generator remembers the
    frame before ``previous`` and feeds it to the interpolator, so from the
    second pair onward the motion path is fitted through three frames (N-2,
    N-1, N) and accelerates/curves with the source instead of assuming a
    constant velocity between N-1 and N.
    """

    def __init__(self, refine: bool = True) -> None:
        self.refine = bool(refine)
        self.generated_frames = 0
        self.held_frames = 0
        self._history: np.ndarray | None = None
        self._interpolator = FrameInterpolator(refine=self.refine)

    def between(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        factor: str | int,
        *,
        scene_cut: bool = False,
    ) -> list[np.ndarray]:
        """Return the synthesised frames that precede ``current``."""
        resolved = resolve_mfg_factor(factor)
        count = resolved - 1
        if count == 0:
            return []
        if previous.shape != current.shape:
            raise ValueError(
                f"frames must share a shape, got {previous.shape} and {current.shape}"
            )
        if scene_cut:
            self.held_frames += count
            self._history = None
            return [np.ascontiguousarray(previous) for _ in range(count)]

        prev2 = self._history
        frames = self._interpolator.between(previous, current, resolved, prev2=prev2)
        if len(frames) != count:
            raise RuntimeError(
                f"MFG produced {len(frames)} frame(s) instead of {count}."
            )
        self.generated_frames += len(frames)
        self._history = previous
        return frames

    @property
    def path(self) -> str:
        if self.generated_frames:
            return self._interpolator.path
        return "scene_cut" if self.held_frames else "not used"

    def describe(self) -> str:
        if self.generated_frames == 0 and self.held_frames:
            return "held frames across scene cuts"
        detail = self._interpolator.describe()
        if self.held_frames:
            return f"{detail}; {self.held_frames} frame(s) held across scene cuts"
        return detail

    def reset_history(self) -> None:
        """Forget N-2, e.g. after a seek or an externally detected scene cut."""
        self._history = None

    def close(self) -> None:
        self._history = None
        self._interpolator.close()

    def __enter__(self) -> "MultiFrameGenerator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
