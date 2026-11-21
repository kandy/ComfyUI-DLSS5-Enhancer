"""Node that synthesises DLSS Multi Frame Generation frames for an IMAGE batch."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar

from ..dlss5.imaging import fit_frame, resize_alpha, rgba_to_tensor, tensor_to_rgba
from ..dlss5.motion import TemporalGuide
from ..dlss5.multiframe import (
    MFG_LABELS,
    MultiFrameGenerator,
    frames_after_mfg,
    resolve_mfg_factor,
)
from ..dlss5.session import DlssSession
from ..dlss5.settings import SessionConfig
from .common import confirm_feature_18, logger, resolve_layout
from .settings_node import DLSS5_SETTINGS


class DLSS5MultiFrameGeneration(io.ComfyNode):
    """Generate up to three frames between every pair of source frames."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5MultiFrameGeneration",
            display_name="DLSS5 Multi Frame Generation",
            category="image/upscaling",
            description=(
                "Synthesises up to three extra frames between consecutive frames "
                "using the NVIDIA Optical Flow Accelerator, refined by the DLSS "
                "neural engine. From the second pair onward it fits a three-frame "
                "trajectory (N-2 -> N-1 -> N) so accelerating motion is followed. "
                "Connect DLSS5 Settings to neural-render (and optionally upscale) "
                "the source frames first; leave it unconnected to only generate "
                "frames. Input order is playback order."
            ),
            search_aliases=[
                "dlss",
                "dlss5",
                "mfg",
                "multi frame generation",
                "frame generation",
                "dlssg",
                "fps",
            ],
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Frames in playback order; the batch order is the temporal order.",
                ),
                DLSS5_SETTINGS.Input(
                    "settings",
                    optional=True,
                    tooltip=(
                        "Optional. When connected, source frames go through DLSS "
                        "neural rendering before frames are generated between them."
                    ),
                ),
                io.Combo.Input(
                    "multiplier",
                    options=MFG_LABELS,
                    default="2x",
                    tooltip=(
                        "Output frames per source frame: 2x inserts one generated "
                        "frame per pair, 4x inserts three (the DLSS MFG maximum)."
                    ),
                ),
                io.Boolean.Input(
                    "neural_refine",
                    default=True,
                    tooltip=(
                        "Run each generated frame through the DLSS neural engine "
                        "(DLAA at 1x) to clean warping artefacts. Turn off for the "
                        "raw optical-flow result."
                    ),
                    advanced=True,
                ),
                io.Boolean.Input(
                    "verify_neural_rendering",
                    default=True,
                    tooltip=(
                        "When settings are connected, fail the render if no signed "
                        "feature-18 execution can be proven. Ignored without settings."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Int.Output("frames"),
                io.Int.Output("generated"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        settings: SessionConfig | None = None,
        multiplier: str = "2x",
        neural_refine: bool = True,
        verify_neural_rendering: bool = True,
    ) -> io.NodeOutput:
        count, height, width, channels = images.shape
        if count == 0:
            raise ValueError("The image batch is empty.")
        factor = resolve_mfg_factor(multiplier)
        keep_alpha = channels == 4
        expected = frames_after_mfg(count, factor)
        progress = ProgressBar(count)

        if settings is None:
            frames = cls._generate_only(
                images, factor, keep_alpha, bool(neural_refine), progress
            )
        else:
            frames, session = cls._enhance_and_generate(
                images, settings, factor, keep_alpha, bool(neural_refine), progress
            )
            confirm_feature_18(session, verify_neural_rendering)

        result = torch.stack(frames)
        if result.shape[0] != expected:
            raise RuntimeError(
                f"MFG produced {result.shape[0]} frames instead of {expected}."
            )
        generated = expected - count
        logger.info(
            "DLSS5: MFG %s generated %d frame(s), %d -> %d total",
            multiplier,
            generated,
            count,
            result.shape[0],
        )
        return io.NodeOutput(result, int(result.shape[0]), int(generated))

    @classmethod
    def _generate_only(
        cls,
        images: torch.Tensor,
        factor: int,
        keep_alpha: bool,
        neural_refine: bool,
        progress: ProgressBar,
    ) -> list[torch.Tensor]:
        frames: list[torch.Tensor] = []
        with MultiFrameGenerator(refine=neural_refine) as generator:
            previous: np.ndarray | None = None
            for index in range(images.shape[0]):
                throw_exception_if_processing_interrupted()
                current = tensor_to_rgba(images[index])
                if previous is not None:
                    for mid in generator.between(previous, current, factor):
                        frames.append(rgba_to_tensor(mid, keep_alpha=keep_alpha))
                frames.append(rgba_to_tensor(current, keep_alpha=keep_alpha))
                previous = current
                progress.update(1)
            logger.info(
                "DLSS5: MFG generated %d frame(s) via %s",
                generator.generated_frames,
                generator.describe(),
            )
        return frames

    @classmethod
    def _enhance_and_generate(
        cls,
        images: torch.Tensor,
        settings: SessionConfig,
        factor: int,
        keep_alpha: bool,
        neural_refine: bool,
        progress: ProgressBar,
    ) -> tuple[list[torch.Tensor], DlssSession]:
        layout = resolve_layout(settings)
        options = settings.options
        count, height, width, _channels = images.shape
        frames: list[torch.Tensor] = []

        with DlssSession(
            layout,
            options,
            input_width=width,
            input_height=height,
            frame_count=count,
        ) as session:
            with MultiFrameGenerator(refine=neural_refine) as generator:
                guide = TemporalGuide(
                    session.render_width,
                    session.render_height,
                    flow_width=options.flow_width,
                    scene_change_threshold=options.scene_change_threshold,
                    enabled=options.wants_motion(count),
                )
                previous: np.ndarray | None = None
                for index in range(count):
                    throw_exception_if_processing_interrupted()
                    rgba = fit_frame(
                        tensor_to_rgba(images[index]),
                        session.render_width,
                        session.render_height,
                    )
                    motion = guide.process(rgba)
                    enhanced, _pts = session.submit(
                        index=index,
                        rgba=rgba,
                        motion=motion.motion,
                        reset=motion.reset,
                        pts=index,
                    )
                    if keep_alpha:
                        enhanced[..., 3] = resize_alpha(
                            rgba[..., 3],
                            session.output_width,
                            session.output_height,
                        )
                    current = np.ascontiguousarray(enhanced)
                    if previous is not None:
                        for mid in generator.between(
                            previous, current, factor, scene_cut=motion.reset
                        ):
                            frames.append(rgba_to_tensor(mid, keep_alpha=keep_alpha))
                    frames.append(rgba_to_tensor(current, keep_alpha=keep_alpha))
                    previous = current
                    progress.update(1)

                logger.info(
                    "DLSS5: MFG %s rendered %d source frame(s) and generated %d "
                    "frame(s) via %s",
                    factor,
                    count,
                    generator.generated_frames + generator.held_frames,
                    generator.describe(),
                )
        return frames, session
