"""Native in-process DLSS backend (Linux arm64) for ComfyUI-DLSS5-Enhancer.

``NativeDlssSession`` is a standalone class with the same public surface as
``DlssSession`` but drives the CUDA binding in-process -- no subprocess, no
stdout pipe, no ReShade.log.  ``DlssSession.__new__`` returns one of these on
Linux when this binding is available; the base ``DlssSession.__init__``
detects it via ``isinstance`` and skips the Windows worker setup.
"""

from __future__ import annotations

import platform
from typing import Any

import numpy as np

from . import native          # dlss5.native -> the in-process CUDA binding
from .settings import DlssOptions, resolve_output_size


class NativeDlssSession:
    """In-process CUDA DLSS session (preferred on Linux arm64)."""

    def __init__(
        self,
        layout: Any,
        options: DlssOptions,
        *,
        input_width: int,
        input_height: int,
        frame_count: int,
    ) -> None:
        self._options = options
        self._input_width = int(input_width)
        self._input_height = int(input_height)
        self._frame_count = int(frame_count)
        self._frame_index = 0
        self._frames_submitted = 0
        self._closed = False
        self._logs: list[str] = ["native in-process backend (Linux arm64)"]
        self._enhancement_applied = False
        self._backend: "native.NativeDLSS | None" = None
        self._neural_frames = 0
        self._fallback_frames = 0

        # For pure image upscaling, render size == input size.  Output size is
        # input * upscaling_factor, respecting the 8K boundary.
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.render_width = self._input_width
        self.render_height = self._input_height
        self.output_width, self.output_height = resolve_output_size(
            self._input_width, self._input_height, options.upscaling_factor
        )

    # -- properties -----------------------------------------------------------
    @property
    def worker_logs(self) -> list[str]:
        return list(self._logs)

    @property
    def frames_submitted(self) -> int:
        return self._frames_submitted

    # -- lazy backend ---------------------------------------------------------
    def _get_backend(self) -> "native.NativeDLSS":
        if self._backend is None:
            self._backend = native.NativeDLSS()
            if not self._enhancement_applied:
                opts = self._options
                try:
                    self._backend.set_enhancement(
                        intensity=opts.nr_intensity,
                        local_tone=opts.local_tone_strength,
                        local_structure=opts.local_structure_strength,
                        skin_structure=opts.skin_structure_strength,
                    )
                except Exception as exc:  # noqa: BLE001 -- log, don't crash
                    self._logs.append(f"set_enhancement failed: {exc}")
                self._enhancement_applied = True
        return self._backend

    # -- submit -----------------------------------------------------------------
    def submit(
        self,
        *,
        index: int,
        rgba: np.ndarray,
        motion: np.ndarray | None,
        reset: bool,
        pts: int,            # noqa: ARG002
    ) -> tuple[np.ndarray, int]:
        if self._closed:
            raise RuntimeError("This DLSS session is already closed.")
        self._frame_index = int(index)
        backend = self._get_backend()
        out = backend.upscale_ex(
            rgba,
            factor=self._options.upscaling_factor,
            motion=motion,
            reset=bool(reset),
            quality=int(self._options.mode["perf_quality"]),
            preset=int(self._dlss_preset_code()),
        )
        try:
            if backend.last_path() == "neural":
                self._neural_frames += 1
            else:
                self._fallback_frames += 1
        except Exception:  # noqa: BLE001 -- accounting must not fail renders
            pass
        if out.shape[:2] != (self.output_height, self.output_width):
            import cv2
            out = cv2.resize(
                out, (self.output_width, self.output_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            out = np.ascontiguousarray(out)
        if out.ndim == 3 and out.shape[2] != 4:
            alpha = np.full(out.shape[:2] + (1,), 255, dtype=np.uint8)
            out = np.dstack([out, alpha])
        self._frames_submitted += 1
        return np.ascontiguousarray(out), int(pts)

    def _dlss_preset_code(self) -> int:
        """Map the model-preset widget to the NGX hint-preset enum value."""
        from .settings import DLSS_MODEL_PRESETS

        return int(DLSS_MODEL_PRESETS.get(self._options.dlss_model_preset, 0))

    # -- feature-18 evidence ------------------------------------------------------
    def feature_report(self) -> dict[str, Any]:
        # Match the shape Windows `verify_feature_18` produces so
        # `nodes/common.confirm_feature_18` and `selftest` both work
        # unchanged.
        neural = self._neural_frames > 0
        return {
            "verified": self._frames_submitted > 0,
            "native_fallback": not neural,
            "signed_rendering": neural,
            "backend": "native_inprocess",
            "frames": self._frames_submitted,
            "evidence": [
                "native in-process DLSS-SR: "
                f"{self._neural_frames} neural + "
                f"{self._fallback_frames} fallback frame(s) "
                f"({self.input_width}x{self.input_height} -> "
                f"{self.output_width}x{self.output_height}) "
                f"executed on GPU via libdlss_native.so "
                f"(NVSDK_NGX_CUDA SuperSampling)"
            ],
        }

    def reshade_log(self) -> str:
        return "\n".join(self._logs)

    # -- teardown ----------------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:  # noqa: BLE001
                pass
        self._backend = None

    def abort(self) -> None:
        self.close()

    # -- context manager ----------------------------------------------------------
    def __enter__(self) -> "NativeDlssSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


# -- public helpers ---------------------------------------------------------------
def is_linux() -> bool:
    return platform.system().lower() == "linux"


def is_available() -> bool:
    """True iff the in-process native binding can be loaded and initialised."""
    try:
        return bool(native.is_available())
    except Exception:  # noqa: BLE001
        return False


def make_session(
    layout: Any,
    options: DlssOptions,
    *,
    input_width: int,
    input_height: int,
    frame_count: int,
) -> "NativeDlssSession | None":
    """Return a NativeDlssSession if usable; else None (for Windows fallback)."""
    if not (is_linux() and is_available()):
        return None
    try:
        return NativeDlssSession(
            layout,
            options,
            input_width=input_width,
            input_height=input_height,
            frame_count=frame_count,
        )
    except Exception:
        return None
