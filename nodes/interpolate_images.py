"""Node that synthesises in-between frames for an IMAGE batch."""

from __future__ import annotations

import torch
from comfy_api.latest import io
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar

from ..dlss5.imaging import rgba_to_tensor, tensor_to_rgba
from ..dlss5.interpolation import (
    INTERPOLATION_FACTORS,
    FrameInterpolator,
    resolve_factor,
)
from .common import logger


class DLSS5InterpolateImages(io.ComfyNode):
    """Insert interpolated frames between consecutive frames of a batch."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5InterpolateImages",
            display_name="DLSS5 Frame Interpolation",
            category="image/upscaling",
            description=(
                "Synthesises in-between frames from a batch of consecutive frames using "
                "the NVIDIA Optical Flow Accelerator, optionally refined by the DLSS "
                "neural engine. Feed frames in playback order; the batch order is the "
                "temporal order. Doubles, triples or quadruples the frame count for "
                "smooth slow motion or higher frame rates."
            ),
            search_aliases=[
                "dlss",
                "dlss5",
                "interpolate",
                "frame interpolation",
                "slow motion",
                "fps",
            ],
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Frames in playback order; the batch order is the temporal order.",
                ),
                io.Combo.Input(
                    "multiplier",
                    options=[label for label in INTERPOLATION_FACTORS if label != "Off"],
                    default="2x",
                    tooltip=(
                        "How many frames the output contains per input frame: 2x adds "
                        "one synthesised frame per pair, 4x adds three."
                    ),
                ),
                io.Boolean.Input(
                    "neural_refine",
                    default=True,
                    tooltip=(
                        "Run each synthesised frame through the DLSS neural engine "
                        "(DLAA at 1x) to clean warping artefacts. Turn off for the raw "
                        "optical-flow result."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Int.Output("frames"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        multiplier: str,
        neural_refine: bool = True,
    ) -> io.NodeOutput:
        count, height, width, channels = images.shape
        if count == 0:
            raise ValueError("The image batch is empty.")
        factor = resolve_factor(multiplier)
        keep_alpha = channels == 4

        if count < 2:
            # Nothing to interpolate; pass the frame through unchanged.
            return io.NodeOutput(images, count)

        interpolator = FrameInterpolator(refine=neural_refine)
        frames: list[torch.Tensor] = []
        try:
            previous = tensor_to_rgba(images[0])
            frames.append(rgba_to_tensor(previous, keep_alpha=keep_alpha))
            progress = ProgressBar(count - 1)
            for index in range(1, count):
                throw_exception_if_processing_interrupted()
                current = tensor_to_rgba(images[index])
                for mid in interpolator.between(previous, current, factor):
                    frames.append(rgba_to_tensor(mid, keep_alpha=keep_alpha))
                frames.append(rgba_to_tensor(current, keep_alpha=keep_alpha))
                previous = current
                progress.update(1)
        finally:
            interpolator.close()

        result = torch.stack(frames)
        logger.info(
            "DLSS5: interpolated %d -> %d frames (%s, %s)",
            count,
            result.shape[0],
            multiplier,
            interpolator.describe(),
        )
        return io.NodeOutput(result, int(result.shape[0]))
