from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from istots.app.convert import (
    ConvertArgumentError,
    ConvertExecutionPlan,
    ConvertPreparationError,
    ConvertRequest,
    execute_convert_plan,
    plan_convert_request,
)
from istots.model_store import DEFAULT_MODEL_ID


class SmokeArgumentError(ValueError):
    pass


class SmokePreparationError(RuntimeError):
    pass


class SmokeCleanupError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeRequest:
    input_sup: Path
    output_dir: Path | None = None
    models_dir: Path | None = None
    max_new_tokens: int = 256
    ocr_mode: str = "default"
    paddle_profile: str = "auto"
    runtime_binary_path: Path | None = None
    paddle_port: int | None = None
    paddle_threads: int | None = None
    paddle_threads_batch: int | None = None
    paddle_gpu_layers: int | None = None
    paddle_no_mmproj_offload: bool = False
    paddle_startup_timeout_sec: float = 120.0
    paddle_ctx_size: int | None = None
    enable_furigana_mask: bool = False
    use_temp_ocr_image_files: bool = True
    no_detector: bool = False
    detector_mode: str = "default"
    detector_family_addon: bool = False
    corrector: str = "off"
    corrector_model_path: Path | None = None
    corrector_mmproj_path: Path | None = None
    qwen_profile: str = "auto"
    qwen_port: int | None = None
    qwen_threads: int | None = None
    qwen_threads_batch: int | None = None
    qwen_gpu_layers: int | None = None
    qwen_no_mmproj_offload: bool = False
    qwen_ctx_size: int | None = None
    qwen_n_predict: int | None = None
    qwen_reasoning: str | None = None
    qwen_startup_timeout_sec: float = 120.0
    corrector_gemini_model: str = "gemini-3.1-pro-preview"
    corrector_api_key_env: str = "GEMINI_API_KEY"
    corrector_thinking_level: str | None = "low"
    corrector_media_resolution: str | None = None
    corrector_cache_dir: Path | None = None
    corrector_gemini_max_attempts: int = 4
    corrector_gemini_request_timeout_sec: float = 90.0
    corrector_gemini_max_workers: int = 3
    srt_policy: str = "safe"
    force: bool = False


@dataclass(frozen=True)
class SmokeExecutionPlan:
    output_dir: Path
    is_auto_output_dir: bool
    convert_plan: ConvertExecutionPlan


@dataclass(frozen=True)
class SmokeExecutionResult:
    output_dir: Path
    is_auto_output_dir: bool
    removed_output_dir: bool
    convert_result: object


def plan_smoke_request(
    request: SmokeRequest,
    *,
    make_tempdir: Callable[[str], str] | None = None,
) -> SmokeExecutionPlan:
    _validate_smoke_request(request)

    is_auto_output_dir = request.output_dir is None
    if request.output_dir is not None:
        output_dir = request.output_dir.expanduser().resolve()
        if output_dir.exists() and not output_dir.is_dir():
            raise SmokeArgumentError("--output-dir must be a directory path")
    else:
        factory = make_tempdir or tempfile.mkdtemp
        output_dir = Path(factory(prefix="istots-smoke-")).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    input_sup = request.input_sup.expanduser().resolve()
    output_srt = output_dir / f"{input_sup.stem}.smoke.srt"
    detector_output = None
    if request.ocr_mode == "default" and not request.no_detector:
        detector_output = output_dir / f"{input_sup.stem}.detector.jsonl"
    corrector_output = None
    if request.corrector != "off":
        corrector_output = output_dir / f"{input_sup.stem}.corrected.jsonl"

    convert_request = ConvertRequest(
        input_sup=input_sup,
        output_srt=output_srt,
        engine="llama-server",
        hf_device="auto",
        hf_dtype="auto",
        model_id=DEFAULT_MODEL_ID,
        models_dir=request.models_dir,
        max_items=None,
        max_new_tokens=request.max_new_tokens,
        ocr_mode=request.ocr_mode,
        paddle_profile=request.paddle_profile,
        runtime_binary_path=request.runtime_binary_path,
        paddle_port=request.paddle_port,
        paddle_threads=request.paddle_threads,
        paddle_threads_batch=request.paddle_threads_batch,
        paddle_gpu_layers=request.paddle_gpu_layers,
        paddle_no_mmproj_offload=request.paddle_no_mmproj_offload,
        paddle_startup_timeout_sec=request.paddle_startup_timeout_sec,
        paddle_ctx_size=request.paddle_ctx_size,
        enable_furigana_mask=request.enable_furigana_mask,
        use_temp_ocr_image_files=request.use_temp_ocr_image_files,
        detector_output=detector_output,
        detector_mode=request.detector_mode,
        detector_family_addon=request.detector_family_addon,
        corrector=request.corrector,
        corrector_output=corrector_output,
        corrector_model_path=request.corrector_model_path,
        corrector_mmproj_path=request.corrector_mmproj_path,
        qwen_profile=request.qwen_profile,
        qwen_port=request.qwen_port,
        qwen_threads=request.qwen_threads,
        qwen_threads_batch=request.qwen_threads_batch,
        qwen_gpu_layers=request.qwen_gpu_layers,
        qwen_no_mmproj_offload=request.qwen_no_mmproj_offload,
        qwen_ctx_size=request.qwen_ctx_size,
        qwen_n_predict=request.qwen_n_predict,
        qwen_reasoning=request.qwen_reasoning,
        qwen_startup_timeout_sec=request.qwen_startup_timeout_sec,
        corrector_gemini_model=request.corrector_gemini_model,
        corrector_api_key_env=request.corrector_api_key_env,
        corrector_thinking_level=request.corrector_thinking_level,
        corrector_media_resolution=request.corrector_media_resolution,
        corrector_cache_dir=request.corrector_cache_dir,
        corrector_gemini_max_attempts=request.corrector_gemini_max_attempts,
        corrector_gemini_request_timeout_sec=request.corrector_gemini_request_timeout_sec,
        corrector_gemini_max_workers=request.corrector_gemini_max_workers,
        srt_policy=request.srt_policy,
        force=request.force,
    )
    try:
        convert_plan = plan_convert_request(convert_request)
    except ConvertArgumentError as exc:
        raise SmokeArgumentError(str(exc)) from exc
    except ConvertPreparationError as exc:
        raise SmokePreparationError(str(exc)) from exc

    return SmokeExecutionPlan(
        output_dir=output_dir,
        is_auto_output_dir=is_auto_output_dir,
        convert_plan=convert_plan,
    )


def execute_smoke_plan(
    plan: SmokeExecutionPlan,
    *,
    verbose: bool = True,
) -> SmokeExecutionResult:
    convert_result = execute_convert_plan(plan.convert_plan, verbose=verbose)
    removed_output_dir = False
    if plan.is_auto_output_dir:
        try:
            shutil.rmtree(plan.output_dir)
        except FileNotFoundError:
            removed_output_dir = False
        except OSError as exc:
            raise SmokeCleanupError(
                f"failed to remove smoke temporary artifacts at {plan.output_dir}: {exc}"
            ) from exc
        else:
            removed_output_dir = True
    return SmokeExecutionResult(
        output_dir=plan.output_dir,
        is_auto_output_dir=plan.is_auto_output_dir,
        removed_output_dir=removed_output_dir,
        convert_result=convert_result,
    )


def _validate_smoke_request(request: SmokeRequest) -> None:
    if request.detector_mode != "default" and request.no_detector and request.corrector == "off":
        raise SmokeArgumentError(
            f"--detector-mode {request.detector_mode} requires detector-enabled smoke validation; "
            "remove --no-detector, keep --ocr-mode default, or enable --corrector"
        )
    if request.detector_family_addon and request.no_detector and request.corrector == "off":
        raise SmokeArgumentError(
            "--detector-family-addon requires detector-enabled smoke validation; "
            "remove --no-detector, keep --ocr-mode default, or enable --corrector"
        )
