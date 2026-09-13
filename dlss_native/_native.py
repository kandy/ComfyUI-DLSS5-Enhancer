"""In-process Python binding for the native Linux DLSS upscaler.

Design choice (FFI vs compiled binding):
    We use Python's built-in :mod:`ctypes` against a small CUDA shared
    library (``libdlss_native.so``) that exposes a flat C ABI.  Advantages:
      * Zero build-time Python dependency (no Cython/pybind11 toolchain).
      * Runs **in-process**: it dlopens a ``.so`` and hands numpy buffers
        straight to the GPU via ``cudaMemcpy`` -- no subprocess, no stdout
        pipe, no fork/copy of a worker.
      * The C ABI is trivially stable and versionable.
    The alternative (pybind11) would add a CMake + Python-dev build step for
    a marginal gain, since the whole API is a handful of functions.

The C library drives the CUDA runtime directly on the local NVIDIA GPU.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

__all__ = [
    "NativeDLSS",
    "upscale",
    "is_available",
    "last_error",
    "supported_scale",
    "last_path",
    "DLSS_OK",
    "SCALE_FACTORS",
    "QUALITY_FOR_FACTOR",
]

# NVSDK_NGX_PerfQuality_Value per ComfyUI scale factor (mirrors the
# DLSS5-Enhancer UPSCALING_MODES perf_quality codes).
QUALITY_FOR_FACTOR = {
    1000: 5,   # 1.0x DLAA
    1500: 2,   # 1.5x Quality
    1724: 1,   # 1.724x Balanced
    2000: 0,   # 2.0x Performance
    3000: 3,   # 3.0x Ultra Performance
}

# ---- C ABI result codes (mirror libdlss_native.cu) -----------------------------
DLSS_OK = 0x000001
DLSS_ERR_BAD_INPUT = ctypes.c_int32(0xBAD00005).value
DLSS_ERR_NOT_INIT = ctypes.c_int32(0xBAD00007).value
DLSS_ERR_CUDA = ctypes.c_int32(0xBAD00002).value
DLSS_ERR_OOM = ctypes.c_int32(0xBAD0000D).value

# Scale factors exposed by the worker (x1000), matching ComfyUI-DLSS5-Enhancer.
SCALE_FACTORS = (1000, 1500, 1724, 2000, 3000)

_LIB_NAME = "libdlss_native.so"


def _find_library() -> Path:
    env = os.environ.get("DLSS_NATIVE_LIB")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"DLSS_NATIVE_LIB={env} is not a file.")
    # Walk up from this package looking for build/libdlss_native.so or a copy.
    here = Path(__file__).resolve().parent
    for base in (here, here / "build", here.parents[1] / "cpp_sample" / "build",
                 Path("/home/kandy/workspace/dlss/dlss_native/build"),
                 Path("/home/kandy/workspace/dlss/dlss_native")):
        cand = base / _LIB_NAME if not base.is_file() else base
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Could not find {_LIB_NAME}. Build it with: "
        "nvcc -O3 -arch=native -shared -o build/libdlss_native.so "
        "src/libdlss_native.cu -lcudart ; or set DLSS_NATIVE_LIB."
    )


_LIB = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = _find_library()
        _LIB = ctypes.CDLL(str(path))
        _setup_proto(_LIB)
    return _LIB


def _setup_proto(L: ctypes.CDLL) -> None:
    L.dlss_init.restype = ctypes.c_int32
    L.dlss_init.argtypes = [ctypes.c_int32]
    L.dlss_shutdown.restype = ctypes.c_int32
    L.dlss_shutdown.argtypes = []
    L.dlss_is_available.restype = ctypes.c_int32
    L.dlss_is_available.argtypes = []
    L.dlss_supported_scale_factor.restype = ctypes.c_int32
    L.dlss_supported_scale_factor.argtypes = [ctypes.c_int32]
    L.dlss_set_enhancement.restype = ctypes.c_int32
    L.dlss_set_enhancement.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    L.dlss_upscale_rgba.restype = ctypes.c_int32
    L.dlss_upscale_rgba.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32,
    ]
    L.dlss_upscale_rgba_ex.restype = ctypes.c_int32
    L.dlss_upscale_rgba_ex.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    L.dlss_last_path.restype = ctypes.c_int32
    L.dlss_last_path.argtypes = []
    L.dlss_get_last_error.restype = ctypes.c_int32
    L.dlss_get_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int32]


def _cstr(L: ctypes.CDLL) -> str:
    buf = ctypes.create_string_buffer(256)
    L.dlss_get_last_error(buf, 256)
    return buf.value.decode("utf-8", "replace")


class DLSSRuntimeError(RuntimeError):
    """Raised when a native DLSS call returns a non-success code."""

    def __init__(self, code: int, message: str = ""):
        self.code = int(code)
        super().__init__(
            f"native DLSS failed (0x{int(code):08X})"
            + (f": {message}" if message else "")
        )


class NativeDLSS:
    """A session over the native CUDA upscaler.

    Usage::

        with NativeDLSS() as dlss:
            out = dlss.upscale(rgba, factor=2.0)   # out: (h*2, w*2, 4) uint8
    """

    def __init__(self, app_id: int = 0xDEADEAF) -> None:
        L = _lib()
        rc = L.dlss_init(app_id)
        if rc != DLSS_OK:
            raise DLSSRuntimeError(rc, _cstr(L))

    def close(self) -> None:
        L = _lib()
        L.dlss_shutdown()

    def __enter__(self) -> "NativeDLSS":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- API ---------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(_lib().dlss_is_available())

    def last_error(self) -> str:
        return _cstr(_lib())

    def supports(self, factor_x1000: int) -> bool:
        return bool(_lib().dlss_supported_scale_factor(int(factor_x1000)))

    def set_enhancement(
        self,
        intensity: float = 2.0,
        local_tone: float = 1.0,
        local_structure: float = 2.0,
        skin_structure: float = -1.0,
    ) -> None:
        L = _lib()
        L.dlss_set_enhancement(
            int(round(intensity * 10)),
            int(round(local_tone * 10)),
            int(round(local_structure * 10)),
            int(round(skin_structure * 10)),
        )

    def last_path(self) -> str:
        """Which engine rendered the last frame: ``"neural"`` or ``"fallback"``."""
        return "neural" if _lib().dlss_last_path() else "fallback"

    def upscale(
        self,
        rgba: np.ndarray,
        factor: float = 2.0,
    ) -> np.ndarray:
        """Upscale an ``HxWx4`` uint8 RGBA array by ``factor`` on the GPU."""
        return self.upscale_ex(rgba, factor=factor)

    def upscale_ex(
        self,
        rgba: np.ndarray,
        factor: float = 2.0,
        motion: np.ndarray | None = None,
        reset: bool = True,
        quality: int | None = None,
        preset: int = 0,
    ) -> np.ndarray:
        """Upscale with full temporal control.

        Args:
            rgba: ``HxWx4`` uint8 input.
            factor: upscale ratio (1.0, 1.5, 1.724, 2.0, 3.0).
            motion: optional ``HxWx2`` float16 backward motion vectors
                (render-resolution pixels); ``None`` means zero motion.
            reset: reset DLSS temporal history before this frame.
            quality: NVSDK perf-quality code (0/1/2/3/5); ``None`` derives
                it from ``factor``.
            preset: DLSS hint preset (0=default, 10/11/12/13=J/K/L/M).
        """
        L = _lib()
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"expected HxWx4 RGBA, got {rgba.shape}")
        if rgba.dtype != np.uint8:
            rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        rgba = np.ascontiguousarray(rgba)
        h, w = rgba.shape[:2]
        fx = int(round(float(factor) * 1000))
        if not L.dlss_supported_scale_factor(fx):
            raise ValueError(f"unsupported scale factor {factor}x")
        if quality is None:
            quality = QUALITY_FOR_FACTOR.get(fx, -1)
        mv_ptr = None
        if motion is not None:
            motion = np.ascontiguousarray(motion, dtype=np.float16)
            if motion.shape[:2] != (h, w) or motion.shape[2] != 2:
                raise ValueError(
                    f"expected HxWx2 motion for {w}x{h} input, got {motion.shape}"
                )
            mv_ptr = ctypes.c_void_p(motion.ctypes.data)
        out_w = int(round(w * float(factor)))
        out_h = int(round(h * float(factor)))
        out = np.empty((out_h, out_w, 4), dtype=np.uint8)
        rc = L.dlss_upscale_rgba_ex(
            ctypes.c_void_p(rgba.ctypes.data), int(w), int(h),
            mv_ptr, int(bool(reset)),
            ctypes.c_void_p(out.ctypes.data), int(out_w), int(out_h),
            fx, int(quality), int(preset),
        )
        if rc != DLSS_OK:
            raise DLSSRuntimeError(rc, _cstr(L))
        return out


# ---- convenience module-level API -------------------------------------------
def is_available() -> bool:
    L = _lib()
    if not L.dlss_is_available():
        L.dlss_init(0xDEADEAF)
    return bool(L.dlss_is_available())


def last_error() -> str:
    return _cstr(_lib())


def supported_scale(factor_x1000: int) -> bool:
    return bool(_lib().dlss_supported_scale_factor(int(factor_x1000)))


def last_path() -> str:
    """Which engine rendered the last frame: ``"neural"`` or ``"fallback"``."""
    return "neural" if _lib().dlss_last_path() else "fallback"


def upscale(rgba: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """One-shot upscale (initialises + runs the CUDA upscaler)."""
    L = _lib()
    if not L.dlss_is_available():
        rc = L.dlss_init(0xDEADEAF)
        if rc != DLSS_OK:
            raise DLSSRuntimeError(rc, _cstr(L))
    return NativeDLSS.__new__(NativeDLSS).upscale(rgba, factor)
