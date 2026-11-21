"""Frame interpolation for the ComfyUI nodes (Linux optical-flow backend).

DLSS Frame Generation (``FrameGeneration.Available``) is a swapchain feature
that is not exposed through ``NVSDK_NGX_CUDA_*`` on Linux arm64 -- probing a
DGX Spark reports it as unavailable. The driver *does* ship the NVIDIA
Optical Flow Accelerator, so this module interpolates with hardware optical
flow (bidirectional flow + consistency-weighted warping) and optionally
refines each synthesised frame with the DLSS neural engine (DLAA at 1x).

If the flow engine cannot be loaded the helper degrades to a linear
cross-fade and says so, rather than failing the render.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("ComfyUI-DLSS5-Enhancer")

INTERPOLATION_CHOICES = ["Off", "2x", "3x", "4x"]
INTERPOLATION_FACTORS = {"Off": 1, "2x": 2, "3x": 3, "4x": 4}
MAX_INTERPOLATION_FACTOR = 4


def resolve_factor(label_or_factor: str | int) -> int:
    """Translate the node's combo label (or a raw multiplier) to a factor."""
    if isinstance(label_or_factor, str):
        try:
            return INTERPOLATION_FACTORS[label_or_factor]
        except KeyError as exc:
            choices = ", ".join(INTERPOLATION_CHOICES)
            raise ValueError(
                f"Unknown frame interpolation {label_or_factor!r}. Choose one of: {choices}."
            ) from exc
    value = int(label_or_factor)
    if value not in INTERPOLATION_FACTORS.values():
        choices = ", ".join(f"{factor}x" for factor in INTERPOLATION_FACTORS.values())
        raise ValueError(f"Unsupported interpolation factor {value}. Choose one of: {choices}.")
    return value


def interpolated_timeline(prev_pts: int, cur_pts: int, factor: int) -> list[int]:
    """Timestamps for the frames written after ``prev_pts`` up to ``cur_pts``.

    ``factor`` divides the source interval into equal slots, so the returned
    list has ``factor`` entries: the ``factor - 1`` synthesised frames plus the
    real current frame, all expressed in the output time base
    (``source_time_base / factor``). The output ends exactly on the last source
    frame, so the total duration matches the source to within one source frame
    (the common ``(N - 1) * factor + 1`` convention).
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    step = int(cur_pts) - int(prev_pts)
    return [int(prev_pts) * factor + step * index for index in range(1, factor)] + [
        int(cur_pts) * factor
    ]


def _crossfade(prev: np.ndarray, next: np.ndarray, t: float) -> np.ndarray:
    blended = prev.astype(np.float32) * (1.0 - t) + next.astype(np.float32) * t
    return np.clip(blended + 0.5, 0.0, 255.0).astype(np.uint8)


class FrameInterpolator:
    """Synthesise intermediate frames between consecutive RGBA8 frames."""

    def __init__(self, refine: bool = True) -> None:
        self.refine = bool(refine)
        self.frames_generated = 0
        self.path = "unavailable"
        self.order = 0
        self._backend = None
        self._failed = False

    # -- backend -------------------------------------------------------------
    def _get_backend(self):
        if self._backend is not None or self._failed:
            return self._backend
        try:
            from . import native

            module = native.load()
            if not module.interp_available():
                raise RuntimeError("NVIDIA Optical Flow Accelerator not available")
            self._backend = module.NativeDLSS()
            self.path = "optical_flow"
        except Exception as exc:  # noqa: BLE001 -- degrade, never fail a render
            logger.warning(
                "DLSS5: frame interpolation falling back to a linear cross-fade (%s).",
                exc,
            )
            self._failed = True
            self._backend = None
            self.path = "blend"
        return self._backend

    # -- API -----------------------------------------------------------------
    def between(
        self,
        prev: np.ndarray,
        next: np.ndarray,
        factor: int,
        prev2: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Return ``factor - 1`` frames strictly between ``prev`` and ``next``.

        When ``prev2`` is supplied the hardware flow path fits a second-order
        trajectory through ``prev2 -> prev -> next`` so acceleration and
        curved motion are followed; without it the pair is interpolated with
        a constant velocity.
        """
        count = max(0, int(factor) - 1)
        if count == 0:
            return []
        if prev.shape != next.shape:
            raise ValueError(
                f"frames must share a shape, got {prev.shape} and {next.shape}"
            )
        backend = self._get_backend()
        if backend is not None:
            try:
                frames = backend.interpolate(
                    prev, next, count=count, refine=self.refine, prev2=prev2
                )
                self.frames_generated += len(frames)
                self.path = "optical_flow" if backend.interp_last_path() == "optical_flow" else "blend"
                order_fn = getattr(backend, "interp_last_order", None)
                self.order = int(order_fn()) if order_fn else (1 if self.path == "optical_flow" else 0)
                return frames
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DLSS5: optical-flow interpolation failed (%s); using a cross-fade.",
                    exc,
                )
                self._backend = None
                self._failed = True
                self.path = "blend"
                self.order = 0
        self.frames_generated += count
        return [
            _crossfade(prev, next, (index + 1) / (count + 1))
            for index in range(count)
        ]

    def describe(self) -> str:
        if self.path == "optical_flow":
            trajectory = (
                "3-frame second-order trajectory"
                if self.order >= 2
                else "pairwise constant-velocity flow"
            )
            return (
                f"NVIDIA Optical Flow Accelerator ({trajectory} + "
                f"consistency-weighted warp{' + DLSS DLAA refine' if self.refine else ''})"
            )
        if self.path == "blend":
            return "linear cross-fade (optical flow unavailable)"
        return "not used"

    def close(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:  # noqa: BLE001
                pass
        self._backend = None
        self.order = 0

    def __enter__(self) -> "FrameInterpolator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
