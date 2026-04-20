from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from istots import model_store, pipeline
from istots.corrector import CorrectorConfig, CorrectorMode
from istots.ocr import PaddleOCRVLRuntimeOverrides, Qwen35RuntimeOverrides

_DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC = 120.0


class ConvertArgumentError(ValueError):
    pass


class ConvertPreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConvertResult:
    output_srt: Path
    processed_count: int
    written_count: int
    device_used: str
    detector_record_count: int = 0
    correction_record_count: int = 0
    correction_applied_count: int = 0
    correction_fallback_count: int = 0


@dataclass(frozen=True)
class ConvertRequest:
    input_sup: Path
    output_srt: Path
    engine: str = "llama-server"
    hf_device: str = "auto"
    hf_dtype: str = "auto"
    model_id: str = model_store.DEFAULT_MODEL_ID
    models_dir: Path | None = None
    max_items: int | None = None
    max_new_tokens: int = 256
    ocr_mode: str = "default"
    paddle_profile: str = "auto"
    runtime_binary_path: Path | None = None
    paddle_port: int | None = None
    paddle_threads: int | None = None
    paddle_threads_batch: int | None = None
    paddle_gpu_layers: int | None = None
    paddle_no_mmproj_offload: bool = False
    paddle_startup_timeout_sec: float = _DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC
    paddle_ctx_size: int | None = None
    enable_furigana_mask: bool = False
    use_temp_ocr_image_files: bool = True
    detector_output: Path | None = None
    detector_mode: str = "default"
    detector_family_addon: bool = False
    corrector: str = "off"
    corrector_output: Path | None = None
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
    qwen_startup_timeout_sec: float = _DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC
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
class ConvertExecutionPlan:
    input_sup: Path
    output_srt: Path
    engine: str
    hf_device: str
    hf_dtype: str
    model_id: str
    models_dir: Path | None
    max_items: int | None
    max_new_tokens: int
    ocr_mode: str
    detector_output: Path | None
    detector_mode: str
    detector_family_addon: bool
    corrector_config: CorrectorConfig | None
    enable_furigana_mask: bool
    srt_policy: str
    runtime_binary_path: Path | None
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides
    use_temp_ocr_image_files: bool
    existing_output_artifacts: tuple[Path, ...]
    resolved_hf_model_path: Path | None = None

    @property
    def corrector_mode(self) -> CorrectorMode | None:
        if self.corrector_config is None:
            return None
        return self.corrector_config.mode

    @property
    def local_files_only(self) -> bool:
        return self.engine == "hf"


def plan_convert_request(request: ConvertRequest) -> ConvertExecutionPlan:
    _validate_convert_request(request)

    input_sup = request.input_sup.expanduser().resolve()
    output_srt = request.output_srt.expanduser().resolve()
    detector_output = (
        request.detector_output.expanduser().resolve()
        if request.detector_output is not None
        else None
    )
    corrector_output = (
        request.corrector_output.expanduser().resolve()
        if request.corrector_output is not None
        else None
    )

    if output_srt.exists() and output_srt.is_dir():
        raise ConvertArgumentError("output_srt must be a file path, not an existing directory")
    if detector_output is not None and detector_output.exists() and detector_output.is_dir():
        raise ConvertArgumentError("detector_output must be a file path, not an existing directory")
    if corrector_output is not None and corrector_output.exists() and corrector_output.is_dir():
        raise ConvertArgumentError("corrector_output must be a file path, not an existing directory")

    _validate_distinct_convert_paths(
        input_sup=input_sup,
        output_srt=output_srt,
        detector_output=detector_output,
        corrector_output=corrector_output,
    )

    corrector_config = _build_corrector_config(
        request=request,
        corrector_output=corrector_output,
    )

    resolved_hf_model_path: Path | None = None
    model_id = request.model_id
    if request.engine == "hf":
        try:
            resolved_hf_model_path = model_store.ensure_local_model(
                model_id=request.model_id,
                models_dir=request.models_dir,
            )
        except Exception as exc:
            raise ConvertPreparationError(f"model check failed: {exc}") from exc
        model_id = str(resolved_hf_model_path)

    existing_output_artifacts = tuple(
        path
        for path in _convert_output_artifacts(
            output_srt=output_srt,
            detector_output=detector_output,
            corrector_output=corrector_output,
        )
        if path.exists()
    )

    return ConvertExecutionPlan(
        input_sup=input_sup,
        output_srt=output_srt,
        engine=request.engine,
        hf_device=request.hf_device,
        hf_dtype=request.hf_dtype,
        model_id=model_id,
        models_dir=request.models_dir,
        max_items=request.max_items,
        max_new_tokens=request.max_new_tokens,
        ocr_mode=request.ocr_mode,
        detector_output=detector_output,
        detector_mode=request.detector_mode,
        detector_family_addon=request.detector_family_addon,
        corrector_config=corrector_config,
        enable_furigana_mask=request.enable_furigana_mask,
        srt_policy=request.srt_policy,
        runtime_binary_path=request.runtime_binary_path,
        paddle_runtime_overrides=PaddleOCRVLRuntimeOverrides(
            profile=request.paddle_profile,
            port=request.paddle_port,
            threads=request.paddle_threads,
            threads_batch=request.paddle_threads_batch,
            gpu_layers=request.paddle_gpu_layers,
            no_mmproj_offload=True if request.paddle_no_mmproj_offload else None,
            startup_timeout_sec=request.paddle_startup_timeout_sec,
            ctx_size=request.paddle_ctx_size,
        ),
        use_temp_ocr_image_files=request.use_temp_ocr_image_files,
        existing_output_artifacts=existing_output_artifacts,
        resolved_hf_model_path=resolved_hf_model_path,
    )


def execute_convert_plan(
    plan: ConvertExecutionPlan,
    *,
    verbose: bool = True,
) -> ConvertResult:
    result = pipeline.convert_sup_to_srt(
        input_sup=plan.input_sup,
        output_srt=plan.output_srt,
        hf_device=plan.hf_device,
        hf_dtype=plan.hf_dtype,
        engine=plan.engine,
        ocr_mode=plan.ocr_mode,
        detector_output=plan.detector_output,
        detector_mode=plan.detector_mode,
        detector_family_addon=plan.detector_family_addon,
        corrector_config=plan.corrector_config,
        model_id=plan.model_id,
        models_dir=plan.models_dir,
        max_items=plan.max_items,
        max_new_tokens=plan.max_new_tokens,
        local_files_only=plan.local_files_only,
        enable_furigana_mask=plan.enable_furigana_mask,
        srt_policy=plan.srt_policy,
        runtime_binary_path=plan.runtime_binary_path,
        paddle_runtime_overrides=plan.paddle_runtime_overrides,
        use_temp_ocr_image_files=plan.use_temp_ocr_image_files,
        verbose=verbose,
    )
    return ConvertResult(
        output_srt=result.output_srt,
        processed_count=getattr(result, "processed_count", getattr(result, "written_count", 0)),
        written_count=getattr(result, "written_count", 0),
        device_used=result.device_used,
        detector_record_count=getattr(result, "detector_record_count", 0),
        correction_record_count=getattr(result, "correction_record_count", 0),
        correction_applied_count=getattr(result, "correction_applied_count", 0),
        correction_fallback_count=getattr(result, "correction_fallback_count", 0),
    )


def _validate_convert_request(request: ConvertRequest) -> None:
    if request.max_items is not None and request.max_items <= 0:
        raise ConvertArgumentError("--max-items must be a positive integer")
    if request.max_new_tokens <= 0:
        raise ConvertArgumentError("--max-new-tokens must be a positive integer")
    if request.engine != "hf":
        if request.hf_device != "auto":
            raise ConvertArgumentError("--hf-device is only valid with --engine hf")
        if request.hf_dtype != "auto":
            raise ConvertArgumentError("--hf-dtype is only valid with --engine hf")
    if request.engine != "llama-server" and _has_paddle_runtime_override_request(request):
        raise ConvertArgumentError("Paddle llama-server overrides are only valid with --engine llama-server")
    if request.detector_output is not None and request.engine != "llama-server":
        raise ConvertArgumentError("--detector-output requires --engine llama-server")
    if request.detector_output is not None and request.ocr_mode != "default":
        raise ConvertArgumentError("--detector-output requires --ocr-mode default")
    if request.detector_mode != "default" and request.engine != "llama-server":
        raise ConvertArgumentError("--detector-mode requires --engine llama-server")
    if request.detector_mode != "default" and request.ocr_mode != "default":
        raise ConvertArgumentError("--detector-mode requires --ocr-mode default")
    if request.detector_mode != "default" and request.detector_output is None and request.corrector == "off":
        raise ConvertArgumentError("--detector-mode requires --detector-output or --corrector")
    if request.detector_family_addon and request.engine != "llama-server":
        raise ConvertArgumentError("--detector-family-addon requires --engine llama-server")
    if request.detector_family_addon and request.ocr_mode != "default":
        raise ConvertArgumentError("--detector-family-addon requires --ocr-mode default")
    if request.detector_family_addon and request.detector_output is None and request.corrector == "off":
        raise ConvertArgumentError("--detector-family-addon requires --detector-output or --corrector")
    if request.corrector != "off" and request.engine != "llama-server":
        raise ConvertArgumentError("--corrector requires --engine llama-server")
    if request.corrector != "off" and request.ocr_mode != "default":
        raise ConvertArgumentError("--corrector requires --ocr-mode default")
    if request.corrector == "off" and request.corrector_output is not None:
        raise ConvertArgumentError("--corrector-output requires --corrector")
    if request.corrector != "qwen-local" and _has_qwen_runtime_override_request(request):
        raise ConvertArgumentError("Qwen llama-server overrides are only valid with --corrector qwen-local")
    if request.corrector == "qwen-local":
        has_model_path = request.corrector_model_path is not None
        has_mmproj_path = request.corrector_mmproj_path is not None
        if has_model_path != has_mmproj_path:
            raise ConvertArgumentError(
                "--corrector qwen-local requires both --corrector-model-path and "
                "--corrector-mmproj-path when either is provided"
            )
    if request.corrector == "gemini":
        if request.corrector_model_path is not None or request.corrector_mmproj_path is not None:
            raise ConvertArgumentError(
                "--corrector-model-path and --corrector-mmproj-path are only valid with --corrector qwen-local"
            )
    if request.corrector_gemini_max_attempts < 1:
        raise ConvertArgumentError("--corrector-gemini-max-attempts must be >= 1")
    if request.corrector_gemini_request_timeout_sec <= 0:
        raise ConvertArgumentError("--corrector-gemini-request-timeout-sec must be > 0")
    if request.corrector_gemini_max_workers < 1:
        raise ConvertArgumentError("--corrector-gemini-max-workers must be >= 1")


def _has_paddle_runtime_override_request(request: ConvertRequest) -> bool:
    return any(
        (
            request.paddle_profile != "auto",
            request.paddle_port is not None,
            request.paddle_threads is not None,
            request.paddle_threads_batch is not None,
            request.paddle_gpu_layers is not None,
            request.paddle_no_mmproj_offload,
            request.paddle_startup_timeout_sec != _DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC,
        )
    )


def _has_qwen_runtime_override_request(request: ConvertRequest) -> bool:
    return any(
        (
            request.qwen_profile != "auto",
            request.qwen_port is not None,
            request.qwen_threads is not None,
            request.qwen_threads_batch is not None,
            request.qwen_gpu_layers is not None,
            request.qwen_no_mmproj_offload,
            request.qwen_ctx_size is not None,
            request.qwen_n_predict is not None,
            request.qwen_reasoning is not None,
            request.qwen_startup_timeout_sec != _DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC,
        )
    )


def _validate_distinct_convert_paths(
    *,
    input_sup: Path,
    output_srt: Path,
    detector_output: Path | None,
    corrector_output: Path | None,
) -> None:
    seen_paths: dict[Path, str] = {}
    for label, path in (
        ("input_sup", input_sup),
        ("output_srt", output_srt),
        ("detector_output", detector_output),
        ("corrector_output", corrector_output),
    ):
        if path is None:
            continue
        previous_label = seen_paths.get(path)
        if previous_label is not None:
            raise ConvertArgumentError(f"{previous_label} and {label} must be different paths")
        seen_paths[path] = label


def _convert_output_artifacts(
    *,
    output_srt: Path,
    detector_output: Path | None,
    corrector_output: Path | None,
) -> tuple[Path, ...]:
    artifacts = [output_srt]
    if detector_output is not None:
        artifacts.append(detector_output)
    if corrector_output is not None:
        artifacts.append(corrector_output)
    return tuple(artifacts)


def _build_corrector_config(
    *,
    request: ConvertRequest,
    corrector_output: Path | None,
) -> CorrectorConfig | None:
    if request.corrector == "off":
        return None

    resolved_corrector_model_path = None
    resolved_corrector_mmproj_path = None
    if request.corrector == "qwen-local":
        if request.corrector_model_path is not None and request.corrector_mmproj_path is not None:
            resolved_corrector_model_path = request.corrector_model_path.expanduser().resolve()
            resolved_corrector_mmproj_path = request.corrector_mmproj_path.expanduser().resolve()
        else:
            try:
                (
                    resolved_corrector_model_path,
                    resolved_corrector_mmproj_path,
                ) = model_store.ensure_local_qwen_corrector_assets(models_dir=request.models_dir)
            except Exception as exc:
                raise ConvertPreparationError(f"corrector asset check failed: {exc}") from exc

    try:
        mode = CorrectorMode(request.corrector)
    except ValueError as exc:
        raise ConvertArgumentError(f"unsupported corrector mode: {request.corrector!r}") from exc

    return CorrectorConfig(
        mode=mode,
        output_path=corrector_output,
        local_model_path=resolved_corrector_model_path,
        local_mmproj_path=resolved_corrector_mmproj_path,
        local_runtime_overrides=Qwen35RuntimeOverrides(
            profile=request.qwen_profile,
            port=request.qwen_port,
            threads=request.qwen_threads,
            threads_batch=request.qwen_threads_batch,
            gpu_layers=request.qwen_gpu_layers,
            no_mmproj_offload=True if request.qwen_no_mmproj_offload else None,
            startup_timeout_sec=request.qwen_startup_timeout_sec,
            ctx_size=request.qwen_ctx_size,
            n_predict=request.qwen_n_predict,
            reasoning=request.qwen_reasoning,
        ),
        api_key_env=request.corrector_api_key_env,
        gemini_model=request.corrector_gemini_model,
        thinking_level=request.corrector_thinking_level,
        media_resolution=request.corrector_media_resolution,
        cache_dir=(
            request.corrector_cache_dir.expanduser().resolve()
            if request.corrector_cache_dir is not None
            else None
        ),
        max_attempts=request.corrector_gemini_max_attempts,
        request_timeout=request.corrector_gemini_request_timeout_sec,
        gemini_max_workers=request.corrector_gemini_max_workers,
    )
