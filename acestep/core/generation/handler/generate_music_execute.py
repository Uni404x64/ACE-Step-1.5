"""Execution helper for ``generate_music`` service invocation with progress tracking."""

import os
import threading
from typing import Any, Dict, List, Optional, Sequence

import torch
from loguru import logger


def _parse_generation_timeout() -> int:
    """Parse ACESTEP_GENERATION_TIMEOUT safely; fall back to 600 on invalid values.

    Two common misconfiguration traps are handled explicitly:
    - A non-numeric value raises ``ValueError`` at module import time if parsed
      naively with ``int(os.environ.get(...))``, crashing the server before any
      generation runs.
    - A value of 0 or negative makes ``Thread.join(timeout=0)`` return
      immediately, so ``is_alive()`` is always ``True`` and every generation
      would time out instantly.
    """
    raw = os.environ.get("ACESTEP_GENERATION_TIMEOUT", "600")
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "ACESTEP_GENERATION_TIMEOUT={!r} is not a valid integer; defaulting to 600s.", raw
        )
        return 600
    if val <= 0:
        logger.warning(
            "ACESTEP_GENERATION_TIMEOUT={} must be positive; defaulting to 600s.", val
        )
        return 600
    return val


# Maximum wall-clock seconds to wait for service_generate before declaring a hang.
# Generous default: most generations finish in 30-120 s, but large batches on slow
# GPUs can take several minutes.  Override via ACESTEP_GENERATION_TIMEOUT env var.
_DEFAULT_GENERATION_TIMEOUT: int = _parse_generation_timeout()


class GenerateMusicExecuteMixin:
    """Run service generation under diffusion progress estimation lifecycle."""

    def _run_generate_music_service_with_progress(
        self,
        progress: Any,
        actual_batch_size: int,
        audio_duration: Optional[float],
        inference_steps: int,
        timesteps: Optional[Sequence[float]],
        service_inputs: Dict[str, Any],
        refer_audios: Optional[List[Any]],
        guidance_scale: float,
        actual_seed_list: Optional[List[int]],
        audio_cover_strength: float,
        cover_noise_strength: float,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        infer_method: str,
    ) -> Dict[str, Any]:
        """Invoke ``service_generate`` while maintaining background progress estimation.

        ``service_generate`` is a blocking CUDA call.  On mid-tier hardware with
        VRAM fragmentation it can hang indefinitely, freezing the Gradio UI.  We
        run it in a daemon thread and enforce ``_DEFAULT_GENERATION_TIMEOUT``
        seconds of wall-clock patience before surfacing a ``TimeoutError``.

        Args:
            progress: Gradio-style progress callback.
            actual_batch_size: Number of audio samples to generate.
            audio_duration: Requested audio length in seconds, or None for default.
            inference_steps: Number of diffusion steps.
            timesteps: Optional custom timestep schedule; overrides ``inference_steps``
                for progress tracking when provided.
            service_inputs: Pre-processed batch tensors and metadata from
                ``_prepare_generate_music_service_inputs``.
            refer_audios: Optional reference audio tensors for conditioning.
            guidance_scale: CFG guidance value forwarded to ``service_generate``.
            actual_seed_list: Per-sample PRNG seeds.
            audio_cover_strength: Cover strength parameter.
            cover_noise_strength: Cover noise strength parameter.
            use_adg: Whether to use adaptive guidance.
            cfg_interval_start: CFG interval start fraction.
            cfg_interval_end: CFG interval end fraction.
            shift: Scheduler shift value.
            infer_method: Diffusion method name (e.g. ``"ode"``).

        Returns:
            Dict with ``"outputs"`` (service_generate return value) and
            ``"infer_steps_for_progress"`` (effective step count used for tracking).

        Raises:
            TimeoutError: when ``service_generate`` exceeds the configured timeout.
            BaseException: any exception raised by ``service_generate`` is re-raised
                transparently so upstream handlers see the original error.
        """
        infer_steps_for_progress = len(timesteps) if timesteps else inference_steps
        progress_desc = f"Generating music (batch size: {actual_batch_size})..."
        progress(0.52, desc=progress_desc)
        stop_event = None
        progress_thread = None
        try:
            stop_event, progress_thread = self._start_diffusion_progress_estimator(
                progress=progress,
                start=0.52,
                end=0.79,
                infer_steps=infer_steps_for_progress,
                batch_size=actual_batch_size,
                duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
                desc=progress_desc,
            )

            _result: Dict[str, Any] = {}
            _error: Dict[str, BaseException] = {}

            def _service_target() -> None:
                try:
                    _result["outputs"] = self.service_generate(
                        captions=service_inputs["captions_batch"],
                        lyrics=service_inputs["lyrics_batch"],
                        metas=service_inputs["metas_batch"],
                        vocal_languages=service_inputs["vocal_languages_batch"],
                        refer_audios=refer_audios,
                        target_wavs=service_inputs["target_wavs_tensor"],
                        infer_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        seed=actual_seed_list,
                        repainting_start=service_inputs["repainting_start_batch"],
                        repainting_end=service_inputs["repainting_end_batch"],
                        instructions=service_inputs["instructions_batch"],
                        audio_cover_strength=audio_cover_strength,
                        cover_noise_strength=cover_noise_strength,
                        use_adg=use_adg,
                        cfg_interval_start=cfg_interval_start,
                        cfg_interval_end=cfg_interval_end,
                        shift=shift,
                        infer_method=infer_method,
                        audio_code_hints=service_inputs["audio_code_hints_batch"],
                        return_intermediate=service_inputs["should_return_intermediate"],
                        timesteps=timesteps,
                    )
                except BaseException as exc:  # noqa: BLE001 — ferry all exceptions across thread boundary
                    _error["exc"] = exc

            gen_thread = threading.Thread(
                target=_service_target, daemon=True, name="service-generate"
            )
            gen_thread.start()
            gen_thread.join(timeout=_DEFAULT_GENERATION_TIMEOUT)

            if gen_thread.is_alive():
                # Attempt to recover VRAM from the stalled CUDA context so that
                # the next generation attempt has a fighting chance.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Count orphaned threads so operators can detect buildup from
                # repeated timeouts without restarting the server.
                stalled = sum(
                    1 for t in threading.enumerate() if t.name == "service-generate"
                )
                logger.error(
                    "[generate_music] service_generate exceeded {}s timeout "
                    "(batch={}, steps={}, duration={:.1f}s, "
                    "orphaned_service_generate_threads={}). "
                    "The CUDA operation may still be running in the background.",
                    _DEFAULT_GENERATION_TIMEOUT,
                    actual_batch_size,
                    inference_steps,
                    audio_duration or 0.0,
                    stalled,
                )
                raise TimeoutError(
                    f"Music generation timed out after {_DEFAULT_GENERATION_TIMEOUT} seconds. "
                    "This usually means the GPU ran out of VRAM or the diffusion loop stalled. "
                    "Try reducing batch size, duration, or inference steps."
                )

            # Re-raise any exception that escaped the worker thread.
            if "exc" in _error:
                raise _error["exc"]

            # Defensive guard: the thread completed without raising but also
            # without populating _result.  This should never happen with
            # except BaseException above, but guard against future refactors.
            if "outputs" not in _result:
                raise RuntimeError(
                    "service_generate completed without producing outputs or raising "
                    "an exception — this is unexpected. Please report this as a bug."
                )

            outputs = _result["outputs"]
        finally:
            if stop_event is not None:
                stop_event.set()
            if progress_thread is not None:
                progress_thread.join(timeout=1.0)
        return {"outputs": outputs, "infer_steps_for_progress": infer_steps_for_progress}
