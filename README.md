# ComfyUI-DLSS5-Enhancer

Run NVIDIA DLSS 5 Neural Rendering over your frames and videos, inside ComfyUI.

DLSS 5 is NVIDIA's neural rendering pass. In a game it reconstructs the material and lighting
detail that real-time rendering has to leave out, such as skin subsurface scattering, light
transmission through hair and contact shadows, guided by the engine's own buffers. This node pack
drives that same renderer over video frames and image batches, with optional 1.5x to 3x
upscaling, optical-flow frame interpolation and multi frame generation.
Without those buffers what it recovers here is material response: skin, hair and fabric structure.

It is not a filter that imitates the look. Frames go to the native renderer and come back
reconstructed.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Nodes](#nodes)
- [Recommended settings](#recommended-settings)
- [Example workflows](#example-workflows)
- [Verification](#verification)
- [Performance](#performance)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Licensing and attribution](#licensing-and-attribution)

## How it works

In games, DLSS 5 runs as "3D-Guided" Neural Rendering: the engine hands the model its rendered
frame together with geometry, texture and lighting buffers, and motion vectors. Video has none of
that. What reaches the model here is the decoded frame plus motion vectors estimated from the
footage itself, so the guidance is weaker than in a game, and the reconstruction is correspondingly
more conservative.

There is no Python API for any of it. The renderer lives in a native D3D12 process that hosts
ReShade and the RenoDX DLSS 5 add-on and evaluates NGX feature 18. This node pack implements the
client side of that worker's binary protocol:

```
  ComfyUI node
      |
      |  RGBA8 frame + FP16 motion vectors + history reset flag
      v
  native D3D12 worker
      (ReShade carrier -> RenoDX DLSS 5 add-on -> NGX feature 18)
      |
      |  RGBA8 reconstructed frame
      v
  ComfyUI node
```

Encoded video carries no motion vectors, so they are estimated per frame with dense optical flow
(OpenCV DIS) and supplied at the render resolution, including automatic history resets on scene
cuts. Everything else is handled here: session setup, dimension negotiation, frame ordering,
cancellation and diagnostics.

The Windows neural components are NVIDIA, ReShade and RenoDX binaries; they are not part of
this repository and are not redistributed by it. The Linux pack does bundle NVIDIA's DLSS SDK
pieces needed to build and run the native backend (`DLSS/`), under NVIDIA's DLSS terms rather
than this project's MIT licence. See [Licensing and attribution](#licensing-and-attribution).

## Requirements

| Item | Requirement |
| --- | --- |
| GPU | NVIDIA ships DLSS 5 for the RTX 50 series. Through this community runtime it also runs on RTX 40 and 30. RTX 20 and older are refused. On Linux (DGX Spark / GB10) the native CUDA backend drives DLSS SuperSampling directly |
| Driver | Current NVIDIA GeForce driver (590+ on Linux for the native backend) |
| ComfyUI | A build with the V3 node API (`comfy_api.latest`) |
| Python | `numpy`, `av` and OpenCV. ComfyUI portable normally ships all three; `opencv-python` is deliberately not in `requirements.txt` so it cannot overwrite an existing `opencv-contrib-python` |
| Linux native backend | CUDA toolkit with `nvcc`, `libnvsdk_ngx.a` from the DLSS SDK, and the driver `libnvidia-ngx.so.1`. No Windows runtime download is needed |

## Installation

All commands below are run from the ComfyUI portable root, the folder containing
`python_embeded` and `ComfyUI`.

### 1. Install the node pack

In ComfyUI-Manager, search for "DLSS5" and install it there. Or clone it by hand:

```bat
git clone https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git ComfyUI\custom_nodes\ComfyUI-DLSS5-Enhancer
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-DLSS5-Enhancer\requirements.txt
```

### 2. Provide the runtime

Nothing is downloaded automatically. Run the setup script once:

```bat
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-DLSS5-Enhancer\install_runtime.py
```

It prints the licensing notice, asks for confirmation, downloads the third-party DLSS 5 Visual
Enhancer release v3.0 from its GitHub releases page (about 467 MB, roughly 700 MB on disk after
extraction) and extracts only its `bin/runtime` and `bin/ffmpeg/bin` contents into the node
folder.

If you already have that application installed, register it instead of downloading again:

```bat
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-DLSS5-Enhancer\install_runtime.py ^
    --runtime-dir "D:\DLSS 5 Visual Enhancer\bin\runtime"
```

Release v3.0 is what this pack is built against; it speaks worker protocol version 4. An older
release fails at session setup with a protocol message.

Additional flags: `--url` overrides the download, `--yes` skips the prompt for unattended
installs, `--keep-archive` keeps the downloaded zip.

**Antivirus:** the worker is an executable that must keep the file name `nvngx.dll`, because the
signed-snippet caller contract checks that name. Running a `.dll` as a process is a pattern
Windows Defender and SmartScreen flag, so the file may be blocked or quarantined. If the node
reports that the worker could not be started, add the `runtime` folder to your exclusion list.

The runtime location is resolved like this:

1. the `runtime_dir` widget on the DLSS5 Settings node. When it is set, nothing else is tried
2. otherwise, in order: the `DLSS5_RUNTIME_DIR` environment variable, `config.json` written by
   `install_runtime.py`, and the bundled `runtime/` folder inside the node pack

ffmpeg and ffprobe are resolved separately and only when the video node runs, before the render
starts, so a missing binary fails immediately rather than after the job: the
`DLSS5_FFMPEG_DIR` environment variable, the `ffmpeg_dir` key in `config.json`, the bundled
`ffmpeg/bin` folder, an `ffmpeg/bin` folder next to the runtime, and finally `PATH`.

### 3. Restart ComfyUI

The nodes appear under **image/upscaling**. If you restart before installing the runtime, they
load normally and only fail when executed, with a message that says how to install it.

### Linux native backend (DGX Spark, no Windows runtime)

On Linux with an NVIDIA GPU, the nodes drive DLSS SuperSampling directly
through the CUDA APIs (`NVSDK_NGX_CUDA_*`), in-process — no ReShade, no
RenoDX, no `nvngx.dll` download. The pack ships the backend
self-contained. The `DLSS/` tree is a git checkout of
https://github.com/NVIDIA/DLSS (the official DLSS SDK): `build.sh`
clones it into `DLSS/` when it is missing, `DLSS_SDK_REPO` overrides
the URL and `DLSS_SDK_DIR` points at an existing tree.

```
ComfyUI-DLSS5-Enhancer/
  dlss_native/            # ctypes binding + prebuilt build/libdlss_native.so
    src/libdlss_native.cu # real DLSS-SR engine, Lanczos fallback
    build.sh              # auto-detects host arch, GPU arch, CUDA toolkit
  DLSS/                   # git checkout of https://github.com/NVIDIA/DLSS
    include/              # NGX SDK headers (rebuild only)
    lib/Linux_x86_64/
      libnvsdk_ngx.a      # NGX static wrapper (rebuild only)
      rel/libnvidia-ngx-dlss.so.310.9.16   # DLSS model plugin (runtime)
    lib/Linux_aarch64/
      libnvsdk_ngx.a      # NGX static wrapper (rebuild only)
      rel/libnvidia-ngx-dlss.so.310.9.16   # DLSS model plugin (runtime)
```

At runtime only `dlss_native/build/libdlss_native.so`, the DLSS model
plugin (`DLSS/lib/Linux_<arch>/rel/libnvidia-ngx-dlss.so`, found
automatically) and the host driver's `libnvidia-ngx.so.1` are needed. To
rebuild from source (e.g. a different CUDA toolkit):

```bash
cd ComfyUI/custom_nodes/ComfyUI-DLSS5-Enhancer/dlss_native && ./build.sh
```

`./build.sh` auto-detects the host arch, the GPU (via `nvidia-smi`) and the
CUDA toolkit; set `ARCH`, `CUDA_HOME` or `DLSS_SDK_DIR` to override the
detected values.

The ComfyUI nodes pick the bundled binding up on Linux with no further
configuration; per-frame motion vectors and history resets flow through
the same `submit()` call as the Windows path. `DLSS_NATIVE_HOME` or
`DLSS_NGX_PLUGIN_DIR` override the discovery if you keep copies
elsewhere. Note the NVIDIA binaries (`libnvidia-ngx-dlss.so`,
`libnvsdk_ngx.a`, headers) are proprietary under NVIDIA's DLSS terms —
redistribute the pack only where those terms permit, exactly as with
the Windows runtime files.

Verify it without ComfyUI:

```bash
python3 ComfyUI/custom_nodes/ComfyUI-DLSS5-Enhancer/selftest.py \
    --frames 5 --mode "2x (Performance)"
```

On Linux this prints `backend: native in-process DLSS-SR` and renders
synthetic frames on the GPU. If NGX is unavailable the binding
transparently falls back to Lanczos and says so in the feature report.

### Frame interpolation (Linux)

Both the video node and `DLSS5 Interpolate Images` can synthesise in-between frames. True DLSS
Frame Generation needs a swapchain and NVIDIA's DLSS-G plugin; neither exists on Linux (the
driver reports `FrameGeneration.Available = 0`), so the pack uses the GPU's NVIDIA Optical Flow
Accelerator instead:

1. `libnvidia-opticalflow.so` computes forward and backward flow between two source frames.
2. Each new frame warps both neighbours along that flow and blends them, weighted by
   forward/backward consistency so occluded pixels follow the frame that can still see them.
3. With `neural_refine` on, the result goes through DLSS SuperSampling at 1x (DLAA) for the
   neural material pass.

So this is optical-flow interpolation with DLSS neural refinement, not DLSS-G. `2x`, `3x` and
`4x` add one, two or three frames per source pair. Everything is GPU-resident and the flow
buffer honours the driver's row pitch. In the video node the output frame rate and time base are
scaled together, so the clip still ends on the last source frame: duration, audio sync,
chapters and metadata are preserved to within one source frame.

**Read this before judging the output:** the interpolator is verified on synthetic translations
(exact shift recovered to RMSE < 1) and on real footage. It cannot invent detail that is absent
from both neighbours. The video node detects scene cuts and holds the outgoing frame across a
cut instead of blending two unrelated shots into a ghost.

### Multi Frame Generation (MFG)

`DLSS5 Multi Frame Generation` follows the DLSS-G MFG contract: from each pair of input frames it
inserts one, two or three synthesised frames (`2x`, `3x`, `4x`). True DLSS-G needs a swapchain, so
the node uses the same optical-flow engine and DLSS refinement described above. With a DLSS5
Settings node connected it first neural-renders (and optionally upscales) the source frames, so the
generated frames match the enhanced output; without Settings it only generates frames and leaves
the source pixels untouched. With Settings connected, scene cuts hold the outgoing frame instead
of blending two unrelated shots.

From the second pair onward the generator also feeds the frame before the pair (N-2) into the
interpolator. Flow is estimated for both N-2 -> N-1 and N-1 -> N, then each pixel follows a
second-order trajectory: velocity is the central difference at N-1 and acceleration is the second
difference. That keeps fast or accelerating motion on its curve instead of assuming a constant
velocity between N-1 and N. The first pair, and any pair after a scene cut, falls back to pairwise
flow because there is no valid N-2. Source frames are assumed to be equally spaced in time.

## Nodes

### DLSS5 Settings

Collects the neural rendering controls once and feeds both processing nodes. Several map onto the
controls NVIDIA documents for game developers: `local_structure_strength` and
`local_tone_strength` are the Structure and Tone intensities, `automatic_mask` is the semantic
masking that tells the model which regions are skin, `skin_structure_strength` is the strength
applied inside that mask, and `dlss_model_preset` selects between models trained with different
weights. The rest are add-on level controls, and some are inert on current runtime builds, which
the table notes.

| Widget | Options or range (default) | Effect |
| --- | --- | --- |
| `upscaling_mode` | `1x (DLAA / native)`, `1.5x (Quality)`, `1.724x (Balanced)`, `2x (Performance)`, `3x (Ultra Performance)` (1x) | 1x enhances at source resolution, the others also upscale |
| `nr_preset` | Default, Preset #1, Preset #2, Preset #3 (Default) | Preset inside the neural rendering model. Measured to have no effect on current runtime builds |
| `nr_style` | Default, Natural, Cinematic (Default) | Natural stays closer to the source, Cinematic pushes contrast |
| `nr_intensity` | 0.00 to 2.00 (1.00) | Strength of the neural pass. Values above 1.00 have no further effect on current builds; below 1.00 blends back towards the source |
| `local_tone_strength` | 0.00 to 2.00 (1.00) | Local tone mapping |
| `local_structure_strength` | 0.00 to 2.00 (1.50) | Local detail and structure reconstruction |
| `skin_structure_strength` | -1.00 to 2.00 (2.00) | Skin and pore reconstruction. Only active while `automatic_mask` is on, which is what locates the skin. `-1` leaves the decision to the model |
| `automatic_mask` | off, on (on) | Let the model detect the regions it treats as skin. Also the gate for `skin_structure_strength` |
| `dlss_model_preset` | Default, J, K, L, M (M) | Forces a specific model. Default, J and K are the softest; L and M reconstruct markedly more skin and hair texture. If the worker reports it applied a different preset, set this back to Default |
| `motion` | auto, optical_flow, none (auto) | Motion vectors for temporal accumulation. `auto` skips them for single images |
| `scene_change_threshold` | 0.01 to 1.00 (0.24) | Mean luminance change above which temporal history resets |
| `warmup_frames` | 0 to 16 (0) | Extra frames the worker renders before the first output settles |
| `runtime_dir` | path (empty) | Overrides runtime discovery |

`motion`, `scene_change_threshold`, `warmup_frames` and `runtime_dir` are marked advanced and are
hidden until advanced widgets are shown.

Output: `DLSS5_SETTINGS`.

### DLSS5 Enhance Images

`IMAGE` in, `IMAGE` out. The batch order is the temporal order, so feed video frames in playback
order for temporal accumulation to work. A four channel input keeps its alpha, which is carried
over rather than reconstructed. The node reports progress and honours the ComfyUI cancel button,
terminating the worker on interruption.

`verify_neural_rendering` (advanced, default on) fails the run when the ReShade log shows no
signed feature-18 execution.

### DLSS5 Interpolate Images

`IMAGE` in, `IMAGE` out, plus the number of frames written. Input order is temporal: with `N`
frames and a factor `F`, the output contains `(N - 1) * F + 1` frames, and each pair contributes
its synthesised frames followed by the real next frame. That is the usual interpolation
convention: the clip ends on the last real frame, so its duration is short by less than one
source frame.

| Input | Default | Notes |
| --- | --- | --- |
| `images` | required | Batch in playback order |
| `multiplier` | 2x | 2x, 3x or 4x |
| `neural_refine` | on | Pass each synthesised frame through 1x DLSS (DLAA) |

A batch of one frame passes through unchanged. Unlike the video node there is no rate metadata
to carry, so a downstream video combiner must use `source_fps * factor`.

### DLSS5 Multi Frame Generation

`IMAGE` in, `IMAGE` out, plus the total frame count and the number of generated frames. Input order
is temporal, and with `N` frames and a factor `F` the output contains `(N - 1) * F + 1` frames.

| Input | Default | Notes |
| --- | --- | --- |
| `images` | required | Batch in playback order |
| `settings` | not connected | Optional. Connected: neural-render (and optionally upscale) the source frames first. Unconnected: only generate frames |
| `multiplier` | 2x | 2x, 3x or 4x. Inserts one, two or three generated frames per pair |
| `neural_refine` | on | Pass each generated frame through 1x DLSS (DLAA) |
| `verify_neural_rendering` | on | Only applies with Settings connected; fails if feature 18 cannot be proven |

This is the MFG contract implemented with the optical-flow backend, not DLSS-G, for the same
swapchain reason as the interpolation node. A batch of one frame is returned unchanged (enhanced
when Settings is connected) because there is no pair to generate between.

### DLSS5 Enhance Video File

File in, file out. Decoding, neural rendering and encoding run concurrently through bounded
queues, so a long video never enters the workflow as one large batch. Original timestamps,
chapters and metadata are preserved. With `copy_audio` on, MKV stream copies audio and subtitles
unchanged, while MP4 and MOV re-encode audio to AAC 192 kbit/s and drop subtitle tracks.

| Input | Default | Notes |
| --- | --- | --- |
| `video_path` | empty | Absolute path to the source file |
| `codec` | HEVC | H.264 and HEVC use NVENC with a libx264/libx265 software fallback. AV1 needs NVENC and has no fallback. ProRes Proxy encodes in software |
| `container` | MKV | MP4, MKV, MOV. ProRes requires MOV or MKV |
| `quality` | Auto | Auto derives a bitrate from resolution and frame rate, Good doubles it, Best quadruples it, Max encodes at constant quality. Ignored for ProRes Proxy |
| `filename_prefix` | DLSS5 | A timestamp is appended, existing files are never overwritten. Path separators, drive letters, control characters and a leading dash are rejected |
| `output_directory` | empty | Empty writes to the ComfyUI output directory. A relative path is resolved inside it and may not escape it; an absolute path is used as given |
| `max_frames` | 0 | 0 renders everything, any other value renders a preview of that many frames, up to 1000000 |
| `copy_audio` | on | Mux the source audio into the result; subtitles only with MKV. Chapters and metadata are carried over either way |
| `verify_neural_rendering` | on (advanced) | Check the ReShade log after the render. The file is written either way |
| `frame_interpolation` | Off | Off, 2x, 3x, 4x. Adds 1, 2 or 3 synthesised frames per source pair and scales the output rate and time base to match |
| `interpolation_neural_refine` | on (advanced) | Run every synthesised frame through 1x DLSS (DLAA). Off is faster and uses the raw flow warps |

Outputs: the written path (`STRING`) and the number of written frames (`INT`), including
interpolated frames.

## Recommended settings

The defaults come from measurements on real generated footage, not from taste. The same clip was
run through the pack with one control changed at a time, comparing the rendered frame against the
source: mean absolute difference, Laplacian detail energy, and high frequency energy as a ringing
indicator.

What that showed on the current runtime build:

- The model reconstructs materials rather than sharpening edges. Overall detail energy drops,
  because generator noise disappears, while skin, hair and fabric gain structure the source only
  implied.
- `dlss_model_preset` matters most. Default, J and K land around 0.56x source detail energy;
  L reaches 0.70x and M 0.73x, which is visible as pores, brow lines and separated beard hair
  instead of waxy skin. M is the default.
- `skin_structure_strength` only works while `automatic_mask` is on. With the mask off, every
  value produces bit identical output; with it on, 2.0 adds a further measurable step of skin
  detail. Both are on by default.
- `nr_intensity` is clamped at 1.0. Values of 1.0, 1.5 and 2.0 produce bit identical output;
  0.0 and 0.5 blend progressively back towards the source.
- `nr_preset` produces bit identical output at every value.
- `local_structure_strength` and `local_tone_strength` work across their full range.
- `nr_style = Cinematic` deepens shadows and shifts the grade, `Natural` softens. Both change the
  look rather than the amount of reconstruction.

Starting points:

| Goal | Settings |
| --- | --- |
| Clean up generated video, same resolution | Defaults: 1x DLAA, model M, mask on, skin 2.0, structure 1.5 |
| Clean up and upscale | Same, with `upscaling_mode = 2x (Performance)` |
| Most realistic faces | Defaults plus `local_structure_strength 2.0` |
| More contrast and punch | `nr_style = Cinematic`, but check the shadows |
| Too strong | `nr_intensity 0.6` to `0.7`, or `local_structure_strength 1.0` |

The neural pass works on what is in the frame. It reconstructs skin and material response, it
does not restore a face that the source no longer contains. For heavily degraded footage, run a
generative restoration pass first and use this node as the final cleanup.

## Example workflows

Enhance a clip inside a workflow, so the frames stay available for compositing. `Load Video` and
`Video Combine` come from
[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite):

```
  VHS Load Video ----+
                     +--> DLSS5 Enhance Images --> VHS Video Combine
  DLSS5 Settings ----+
```

Enhance a long file, holding only a few frames in memory and keeping the audio:

```
  DLSS5 Settings --> DLSS5 Enhance Video File --> output path
```

Still images work with any `IMAGE` source. With `motion = auto`, a single image is rendered
without motion vectors automatically.

## Verification

Both processing nodes inspect the ReShade log after rendering and fail if signed feature 18
execution cannot be proven. This matters, because a silent fallback to plain upscaling would be
indistinguishable from success in the output itself. The check is controlled by
`verify_neural_rendering` and runs once the worker has exited, since ReShade rewrites its log on
start and flushes it on exit. The video node writes and muxes its file before verifying, so a
failed check never discards a finished render. The same applies to the worker itself: if it exits
badly after every frame was rendered, the node logs a warning and keeps the file. The one case
that reports a failure on an otherwise good render is a worker that has to be killed on a close
timeout, because it never flushes its log.

On the Linux native backend there is no ReShade log; instead the feature
report records how many frames ran on the neural engine
(`N neural + M fallback frame(s) ... (NVSDK_NGX_CUDA SuperSampling)`).
A nonzero fallback count means those frames were plain Lanczos upscales.

Two log observations are normal and not errors (Windows worker only):

- `NR upscaling fell back to native`: current runtime builds decline the low resolution colour
  contract, so DLSS upscales first and the neural pass then runs at the output resolution. The
  frames are still upscaled and neurally rendered.
- `vtable::Hook(Failed to find NVSDK_NGX_D3D12_CreateFeature)`: the add-on probes hook points it
  does not always need. Successful renders contain these lines too.

Give the worker the GPU. Another process rendering at the same time, such as a second ComfyUI
queue item or a game, can keep the signed runtime from initialising, which surfaces as a failed
verification.

The protocol and runtime can also be exercised without ComfyUI:

```bat
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-DLSS5-Enhancer\selftest.py ^
    --frames 5 --mode "2x (Performance)"
```

This starts the worker, performs the handshake, sends five synthetic frames and prints the
negotiated render and output sizes plus the feature 18 evidence. It is the fastest way to tell a
runtime or driver problem apart from a workflow problem. It also accepts `--width`, `--height`
and `--runtime-dir`.

## Performance

- The worker starts once per render, not once per frame. Prefer one long batch over many short
  ones.
- Frames are streamed to the worker one at a time, but DLSS5 Enhance Images allocates the whole
  output batch up front at the upscaled resolution. Peak memory is roughly the input batch plus
  the batch times the squared upscaling factor. For long videos use DLSS5 Enhance Video File,
  which never materialises the clip.
- Optical flow is computed at a width of at most 640 pixels and scaled up, which keeps the guide
  cost small next to the neural pass. `motion = none` skips it, which suits a static camera, but
  it is not a way to process unrelated images together; see
  [Known limitations](#known-limitations).
- The output long edge is capped at 7680 pixels and the short edge at 4320, in either
  orientation. The node names a usable mode when a request exceeds that, or says so when even 1x
  does not fit.

One measurement, RTX 5090 with driver 610.74, 640x360 to 1280x720 at 2x Performance: about
4 seconds of worker startup, then roughly 10 frames per second.

## Known limitations

- HDR sources are converted to 8-bit SDR without a tone-mapping operator, so highlights clip
  and the grade shifts. Convert HDR footage to SDR yourself first for a predictable result.
  The encoder writes `yuv420p`, or `yuv422p10le` for ProRes Proxy. The video node logs a
  warning when it detects an HDR transfer function.
- Sources without a frame count in their metadata, common for MKV, WebM and variable frame rate
  files, trigger a full counting pass with ffprobe before rendering starts. The node logs a line
  before doing so, but the queue looks idle while it runs.
- Only one worker runs at a time per render. Two DLSS nodes executing in parallel will compete
  for the GPU.
- `motion = none` sends zero motion and resets the temporal history once, which assumes the
  batch is one coherent sequence. For a batch of unrelated images, run them separately.
- `nb_frames` is an estimate in several containers. When the rendered frame count disagrees with
  a count taken from metadata, the node warns and keeps the render; only a count from the exact
  ffprobe counting pass is treated as authoritative and fails the run.
- Running the video node twice with identical inputs replays the cached result and writes no
  new file. Change an input, or the source file, to render again.
- Windows only, except for the native Linux CUDA backend: on Linux the
  nodes run DLSS SuperSampling in-process (no ReShade/RenoDX worker), so
  neural-rendering-specific controls (`nr_preset`, `nr_style`,
  `automatic_mask`, ReShade-log verification) do not apply there; motion
  vectors, history resets, upscaling modes and model presets do.

## Troubleshooting

| Message | Cause and fix |
| --- | --- |
| `No DLSS 5 runtime was found` | Runtime not installed or the path is wrong. Run `install_runtime.py` or set `DLSS5_RUNTIME_DIR` |
| `The DLSS 5 runtime in ... is incomplete` | Files were deleted or extracted partially. The message lists what is missing |
| `The native DLSS worker at ... could not be started` | Antivirus is blocking the worker, or the file is not executable. Add the runtime folder to the exclusion list |
| `nvidia-smi is unavailable` | The NVIDIA driver tools are not on `PATH`, or no NVIDIA GPU is present |
| `is outside the supported RTX 30/40/50 scope` | An RTX card older than the 30 series |
| `No supported NVIDIA RTX GPU was detected` | nvidia-smi ran but reported no card with RTX in its name |
| `OpenCV is required` | Install `opencv-python`, or `opencv-contrib-python` if you already use it |
| `The PyAV package is required` | Only the video node needs it: `python_embeded\python.exe -m pip install av` |
| `needs the tested experimental Ampere pair` | RTX 30 only runs with the verified RenoDX 4.70 and DLSS NR 310.8.SF-v2 files, checked by SHA-256. The message prints the hashes actually installed |
| `does not speak the version-4 protocol` | The registered runtime is from a different release. Use v3.0 |
| `ffmpeg/ffprobe were not found` | Only the video node needs them. Set `DLSS5_FFMPEG_DIR` or put them on `PATH` |
| `native binding ... not available` (Linux) | `dlss_native/build/libdlss_native.so` is missing or failed to load. Run `dlss_native/build.sh`; needs `nvcc`, the DLSS SDK tree and the NVIDIA driver |
| frames fell back to Lanczos (Linux) | NGX initialisation failed (driver too old, no supported GPU, or DLSS plugin not found). Check the console log; `DLSS_NGX_PLUGIN_DIR` can point at a directory containing `libnvidia-ngx-dlss.so` |
| `DLSS <mode> is unavailable for ...` | The driver rejected that output size. Choose a lower upscaling mode |
| `applied DLSS model preset X instead of the requested Y` | The runtime does not support that model. Set `dlss_model_preset` to Default |
| `crashed inside feature-18 evaluation (access violation)` | Driver and RenoDX or DLSS NR versions do not match. Update the driver and reinstall the runtime |
| `feature-18 execution was not verified` | The worker ran but produced no neural rendering evidence. Most often another process was using the GPU. Run it again with nothing else rendering |
| `AV1 NVENC cannot encode ...` | The GPU or driver cannot encode that resolution in AV1. Use H.264 or HEVC |

## Project layout

```
ComfyUI-DLSS5-Enhancer/
  install_runtime.py     Download or register the native runtime
  selftest.py            Protocol smoke test without ComfyUI
  dlss5/
    protocol.py          Binary wire format (version 4)
    session.py           Worker lifecycle, handshake, frame streaming
    motion.py            Optical flow temporal guides and scene cut resets
    settings.py          Public controls and their native translation
    paths.py             Runtime discovery and validation
    diagnostics.py       GPU checks, feature 18 proof, failure analysis
    imaging.py           IMAGE tensor and RGBA8 conversion, letterboxing
    media.py             Probing, decoding, NVENC encoding, muxing
    interpolation.py     Optical-flow interpolation + DLSS refine, timeline maths
    multiframe.py        Multi Frame Generation contract over the interpolation engine
    selftest.py          The smoke test itself
  nodes/                 The five ComfyUI nodes
  tests/                 Helper tests that need no GPU or ComfyUI
  config.json            Written by install_runtime.py, not tracked
  runtime/               Native runtime, not tracked
  ffmpeg/                Bundled ffmpeg, not tracked
```

To uninstall, delete the node folder. The runtime, `config.json` and any bundled ffmpeg live
inside it, so nothing is left behind.

## Licensing and attribution

The source code in this repository is MIT licensed. See [LICENSE](LICENSE).

The runtime it drives is not. The Windows runtime is NVIDIA, ReShade and RenoDX components plus
the upstream project's own worker executable, each under its own terms: NVIDIA's DLSS and neural
rendering libraries are proprietary, ReShade is BSD-3-Clause, the RenoDX DLSS 5 add-on carries its
own distribution terms, and the worker is built and licensed by the upstream project. None of that
Windows runtime is included here.

The Linux pack is self-contained: the `DLSS/` tree is a git checkout of the
official DLSS SDK repo https://github.com/NVIDIA/DLSS. The NVIDIA
material it carries — `DLSS/include/` (NGX SDK headers), the
`libnvsdk_ngx.a` static wrappers and the `rel/libnvidia-ngx-dlss.so`
model plugins — is under NVIDIA's DLSS terms, not this project's MIT
licence. Redistribute the pack only where those terms allow it. The
`nvofapi` headers are MIT (from dxvk-nvapi).

The worker protocol was reimplemented for this pack. The runtime and the original application come
from [Merserk/dlss5-visual-enhancer](https://github.com/Merserk/dlss5-visual-enhancer), which
makes NVIDIA's renderer usable outside a game engine. This project is not affiliated with or
endorsed by NVIDIA, ReShade, RenoDX or that project.
