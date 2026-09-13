"""Load the in-process native DLSS binding (Linux arm64).

The binding lives in the sibling ``dlss_native`` package inside the parent
workspace (``/home/kandy/workspace/dlss/dlss_native``).  We add that parent
directory to ``sys.path`` (idempotently) so ComfyUI plugins can be installed
anywhere on the disk while still finding the native CUDA shared library.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _candidate_dirs() -> list[str]:
    # `here` is the dlss5/ subpackage of the ComfyUI plugin.
    here = Path(__file__).resolve().parent          # .../ComfyUI-DLSS5-Enhancer/dlss5
    plugin_root = here.parent                        # .../ComfyUI-DLSS5-Enhancer
    # 1) the binding bundled inside the distributed pack, then
    # 2) the dev-workspace layout (plugin_root -> workspace), then
    # 3) a manual override.
    workspace = plugin_root.parent                   # .../dlss
    dirs = [str(plugin_root)]
    dirs.append(str(workspace))
    for parent in workspace.parents[:3]:
        dirs.append(str(parent))
    env = os.environ.get("DLSS_NATIVE_HOME")
    if env:
        dirs.append(env)
    return dirs


def _ensure_importable() -> None:
    for d in _candidate_dirs():
        if d and Path(d, "dlss_native", "__init__.py").is_file() and d not in sys.path:
            sys.path.insert(0, d)


def load() -> ModuleType:
    """Import :mod:`dlss_native` and return it.

    Raises ``NativeBindingUnavailable`` (ImportError) when the binding cannot
    be found or failed to dlopen -- callers can then fall back to the
    Windows pipe-worker transport if they still have it.
    """
    _ensure_importable()
    try:
        return importlib.import_module("dlss_native")
    except Exception as exc:  # noqa: BLE001 -- surface as one clear error
        raise ImportError(
            "The native in-process DLSS binding is not available on this platform. "
            "Expected the 'dlss_native' package with build/libdlss_native.so. "
            f"Last error: {exc!r}"
        ) from exc


class NativeBindingUnavailable(ImportError):
    pass


# A tiny convenience re-export so downstream code can do `from dlss5.native import NativeDLSS`.
_native: ModuleType | None = None


def _ensure() -> ModuleType:
    global _native
    if _native is None:
        _native = load()
    return _native


def __getattr__(name: str):  # PEP 562 -- lazy attribute forwarding
    return getattr(_ensure(), name)
