"""Helpers shared by the DLSS 5 processing nodes."""

from __future__ import annotations

import logging
from pathlib import Path

from ..dlss5.paths import RuntimeLayout, RuntimeMissing, find_runtime
from ..dlss5.session import DlssSession
from ..dlss5.settings import SessionConfig

logger = logging.getLogger("ComfyUI-DLSS5-Enhancer")


def resolve_layout(config: SessionConfig) -> RuntimeLayout:
    """Locate the runtime, or a placeholder when the native backend suffices.

    On Linux the in-process binding needs no Windows runtime at all. The video
    node still needs ffmpeg, which ``RuntimeLayout`` resolves lazily from
    PATH, so an empty root is enough there.
    """
    try:
        return find_runtime(config.runtime_override)
    except RuntimeMissing:
        try:
            from ..dlss5 import native_backend

            if native_backend.is_linux() and native_backend.is_available():
                logger.info(
                    "DLSS5: using the native in-process backend; no Windows "
                    "runtime folder is required on Linux."
                )
                return RuntimeLayout(
                    root=Path(config.runtime_override or "."), config={}
                )
        except Exception:  # noqa: BLE001 -- fall through to the original error
            pass
        raise


def confirm_feature_18(session: DlssSession, enabled: bool) -> None:
    """Fail loudly when the render produced no signed feature-18 evidence.

    Without this the frames could come back merely upscaled, which is
    indistinguishable from success in the output itself.
    """
    if not enabled:
        return
    report = session.feature_report()
    if report["native_fallback"] and report.get("backend") == "native_inprocess":
        # Linux native backend: fallback means plain Lanczos upscaling with
        # no neural pass (e.g. NGX unavailable). Loud, but non-fatal.
        logger.warning(
            "DLSS5: %d frame(s) fell back to Lanczos upscaling without "
            "neural rendering; check the NGX driver installation.",
            session.frames_submitted,
        )
    elif report["native_fallback"]:
        # Expected on current runtime builds: DLSS upscales first, then the
        # neural pass runs at the output resolution instead of inside the
        # upscaler. Neural rendering still happened.
        logger.info(
            "DLSS5: neural rendering ran on the native path after upscaling "
            "(the signed runtime declined the low-resolution colour contract)."
        )
    logger.info(
        "DLSS5: feature-18 neural rendering verified for %d frame(s).",
        session.frames_submitted,
    )
