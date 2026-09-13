"""Lifecycle of one DLSS 5 neural-rendering worker.

Two transports are supported:

  * **Native in-process (Linux arm64)** -- ``NativeDlssSession`` from
    :mod:`dlss5.native_backend` drives the CUDA binding in-process.  On Linux,
    ``DlssSession.__new__`` returns one of these, and the base ``__init__``
    detects it via ``isinstance`` and skips the Windows worker setup.
  * **Windows pipe-worker (fallback)** -- the original ``nvngx.dll``
    executable running ``--video`` on the ReShade carrier stack.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, NoReturn

import numpy as np

from . import protocol
from .diagnostics import (
    LogRing,
    describe_worker_failure,
    drain,
    ensure_supported,
    relevant_lines,
    verify_feature_18,
)
from .paths import RuntimeLayout
from .settings import DLSS_MODEL_PRESETS, DlssOptions, resolve_output_size


def _preset_name(value: int) -> str:
    """Name a model preset the way the widget spells it."""
    for name, number in DLSS_MODEL_PRESETS.items():
        if number == value:
            return f"{name} ({value})"
    return str(value)


class DlssSession:
    """A frame stream against the DLSS native renderer."""

    # ---- factory ------------------------------------------------------------
    def __new__(cls, layout, options, *, input_width, input_height, frame_count):
        """Factory: on Linux with the in-process CUDA binding available, construct
        and return a ``NativeDlssSession`` (standalone class).  Because it is not
        a subclass of ``DlssSession``, Python's ``type.__call__`` sees an
        instance of a non-subclass and skips calling ``DlssSession.__init__``;
        the native object is already fully initialised here."""
        try:
            from .native_backend import is_linux, is_available, NativeDlssSession
            if is_linux() and is_available():
                return NativeDlssSession(
                    layout,
                    options,
                    input_width=input_width,
                    input_height=input_height,
                    frame_count=frame_count,
                )
        except Exception:
            pass  # fall through to Windows pipe-worker
        return object.__new__(cls)

    # ---- init ---------------------------------------------------------------
    def __init__(
        self,
        layout: RuntimeLayout,
        options: DlssOptions,
        *,
        input_width: int,
        input_height: int,
        frame_count: int,
    ) -> None:
        # Skip the Windows worker path when __new__ handed us the native
        # class -- it runs its own __init__ (in native_backend.py).  Lazy
        # import so we don't add a top-level circular import on module load.
        try:
            from .native_backend import NativeDlssSession
            if isinstance(self, NativeDlssSession):
                return
        except Exception:
            pass
        self._init_worker(
            layout,
            options,
            input_width=input_width,
            input_height=input_height,
            frame_count=frame_count,
        )

    # ------------------------------------------------------------------------
    # Windows pipe-worker path (only used when native backend is unavailable)
    # ------------------------------------------------------------------------

    def _init_worker(
        self,
        layout: RuntimeLayout,
        options: DlssOptions,
        *,
        input_width: int,
        input_height: int,
        frame_count: int,
    ) -> None:
        self.layout = layout
        self.options = options
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.gpu, self.bundle = ensure_supported(layout)
        self.output_width, self.output_height = resolve_output_size(
            self.input_width, self.input_height, options.upscaling_factor
        )

        self._logs = LogRing()
        self._log_thread: threading.Thread | None = None
        self._closed = False
        self._frames_submitted = 0
        self._started_at = time.time() - 1.0

        try:
            self._worker = subprocess.Popen(
                [str(layout.worker), "--video"],
                cwd=str(layout.root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise RuntimeError(
                f"The native DLSS worker at {layout.worker} could not be started: {exc}. "
                "It is an executable despite the .dll name, so antivirus software often "
                "quarantines or blocks it. Add the runtime folder to the exclusion list."
            ) from exc

        try:
            self._log_thread = threading.Thread(
                target=drain,
                args=(self._worker.stderr, self._logs),
                name="dlss5-worker-log",
                daemon=True,
            )
            self._log_thread.start()
            self.setup = self._handshake(frame_count)
        except BaseException:
            self.abort()
            raise

    def _handshake(self, frame_count: int) -> protocol.SetupResponse:
        native = self.options.native()
        header = protocol.pack_video_header(
            input_width=self.input_width,
            input_height=self.input_height,
            output_width=self.output_width,
            output_height=self.output_height,
            warmup_frames=max(0, int(self.options.warmup_frames)),
            frame_count=max(1, int(frame_count)),
            perf_quality=int(self.options.mode["perf_quality"]),
            native=native,
        )
        try:
            self._worker.stdin.write(header)
            self._worker.stdin.flush()
            payload = protocol.read_exact(
                self._worker.stdout, protocol.SETUP_RESPONSE.size
            )
        except (EOFError, BrokenPipeError, OSError) as exc:
            try:
                code = self._worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                code = None
            self._join_log_thread()
            details = "\n".join(self.worker_logs[-60:]) or "The worker produced no output."
            raise RuntimeError(
                "The native worker failed during DLSS setup or does not speak the "
                f"version-4 protocol (exit {code}):\n{details}"
            ) from exc

        setup = protocol.SetupResponse.unpack(payload)
        mode_name = self.options.mode["name"]
        if not setup.ok:
            details = "\n".join(self.worker_logs[-40:])
            raise RuntimeError(
                f"DLSS {mode_name} is unavailable for "
                f"{self.output_width}x{self.output_height} (NGX 0x{setup.result:08X}). "
                "Pick a lower upscaling mode or update the NVIDIA driver."
                + (f"\n{details}" if details else "")
            )
        if (setup.output_width, setup.output_height) != (self.output_width, self.output_height):
            raise RuntimeError(
                "The native worker negotiated "
                f"{setup.output_width}x{setup.output_height} instead of the requested "
                f"{self.output_width}x{self.output_height}."
            )
        requested_preset = int(native["dlss_model_preset"])
        if setup.applied_model_preset != requested_preset:
            raise RuntimeError(
                "The worker applied DLSS model preset "
                f"{_preset_name(setup.applied_model_preset)} instead of the requested "
                f"{self.options.dlss_model_preset}. This runtime does not support that "
                "model. Set dlss_model_preset to Default."
            )
        if setup.render_width < 64 or setup.render_height < 64:
            raise RuntimeError(
                f"DLSS returned an unusable render size {setup.render_width}x"
                f"{setup.render_height}; both edges must be at least 64 pixels."
            )
        return setup

    # -- properties ------------------------------------------------------------
    @property
    def render_width(self) -> int:
        return self.setup.render_width

    @property
    def render_height(self) -> int:
        return self.setup.render_height

    @property
    def worker_logs(self) -> list[str]:
        return self._logs.snapshot()

    @property
    def frames_submitted(self) -> int:
        return self._frames_submitted

    def reshade_log(self) -> str:
        """Return this worker's ReShade output (empty string on the native path)."""
        path = self.layout.reshade_log
        try:
            if path.stat().st_mtime < self._started_at:
                return ""
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def feature_report(self) -> dict[str, Any]:
        if not self._closed:
            raise RuntimeError(
                "The feature-18 report is only available after the session is closed."
            )
        return verify_feature_18(self.reshade_log())

    # -- streaming -------------------------------------------------------------
    def _raise_worker_failure(self, index: int, cause: BaseException) -> NoReturn:
        try:
            exit_code = self._worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            exit_code = None
        self._join_log_thread()
        raise RuntimeError(
            describe_worker_failure(
                exit_code=exit_code,
                frame_index=index,
                worker_logs=self.worker_logs,
                reshade_lines=relevant_lines(self.reshade_log(), limit=60),
                gpu=self.gpu,
                bundle=self.bundle,
            )
        ) from cause

    def submit(
        self,
        *,
        index: int,
        rgba: np.ndarray,
        motion: np.ndarray,
        reset: bool,
        pts: int,
    ) -> tuple[np.ndarray, int]:
        """Send one render-resolution RGBA frame and return the enhanced frame."""
        if self._closed:
            raise RuntimeError("This DLSS session is already closed.")
        stdin, stdout = self._worker.stdin, self._worker.stdout
        try:
            stdin.write(protocol.pack_frame_header(index, reset, pts))
            stdin.write(protocol.payload_bytes(rgba, np.dtype(np.uint8)))
            stdin.write(protocol.payload_bytes(motion, np.dtype(np.float16)))
            stdin.flush()
            header = protocol.ResultHeader.unpack(
                protocol.read_exact(stdout, protocol.RESULT_HEADER.size)
            )
        except (EOFError, BrokenPipeError, OSError, RuntimeError) as exc:
            self._raise_worker_failure(index, exc)

        expected = self.output_width * self.output_height * 4
        if not header.ok or header.index != index or header.byte_count != expected:
            raise RuntimeError(
                f"The worker returned an invalid response for frame {index} "
                f"(index {header.index}, {header.byte_count} of {expected} bytes)."
            )
        if header.ngx_result != 1:
            raise RuntimeError(
                f"Feature-18 evaluation failed on frame {index}: 0x{header.ngx_result:08X}"
            )

        output = np.empty((self.output_height, self.output_width, 4), dtype=np.uint8)
        try:
            protocol.read_into(stdout, output)
        except (EOFError, OSError) as exc:
            self._raise_worker_failure(index, exc)
        self._frames_submitted += 1
        return output, header.pts

    # -- teardown ----------------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        if self._worker.stdin and not self._worker.stdin.closed:
            self._worker.stdin.close()
        try:
            exit_code = self._worker.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            self.abort()
            raise RuntimeError(
                "The native DLSS worker did not exit within 60 seconds and was killed."
            ) from exc
        self._closed = True
        self._close_pipes()
        self._join_log_thread()
        if exit_code:
            raise RuntimeError(
                f"The native DLSS worker exited with code {exit_code}:\n"
                + "\n".join(self.worker_logs[-40:])
            )

    def _join_log_thread(self, timeout: float = 2.0) -> None:
        if self._log_thread is not None and self._log_thread.ident is not None:
            self._log_thread.join(timeout=timeout)

    def _close_pipes(self) -> None:
        for pipe in (self._worker.stdin, self._worker.stdout):
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except OSError:
                    pass

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker.poll() is None:
            try:
                self._worker.terminate()
                self._worker.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self._worker.kill()
                except OSError:
                    pass
        self._close_pipes()
        self._join_log_thread()

    # -- context manager ----------------------------------------------------------
    def __enter__(self) -> "DlssSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            try:
                self.close()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()
