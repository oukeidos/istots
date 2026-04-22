from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable

from istots import model_store, pipeline
from istots.corrector import CorrectorConfig, CorrectorMode
from istots.gemini_auth import GeminiAuthError, require_gemini_api_key
from istots.ocr import PaddleOCRVLRuntimeOverrides, Qwen35RuntimeOverrides

_DEFAULT_RUNTIME_STARTUP_TIMEOUT_SEC = 120.0
_ESTIMATED_BYTES_PER_ROW = 22_000.0
_DEFAULT_FAST_ROW_SEC = 1.20
_DEFAULT_TALL_ROW_SEC = 1.45
_DEFAULT_DEFAULT_ROW_SEC = 1.35
_DEFAULT_BACKEND_LOAD_SEC = 7.0
_CPU_FAST_TO_TALL_RATE_MULTIPLIER = 1.60
_CPU_FAST_TO_TALL_BLEND_UNIQUES = 8


class ConvertArgumentError(ValueError):
    pass


class ConvertPreparationError(RuntimeError):
    pass


ConvertProgressEvent = pipeline.ConversionProgressEvent


@dataclass(frozen=True)
class ConvertProgressSnapshot:
    phase: str
    headline: str
    detail: str
    fraction: float
    elapsed_sec: float
    eta_sec: float | None


@dataclass
class _BranchProgressState:
    total_rows: int = 0
    processed_rows: int = 0
    total_unique: int | None = None
    processed_unique: int = 0
    backend_load_started_elapsed: float | None = None
    backend_load_elapsed: float | None = None
    ocr_started_elapsed: float | None = None
    last_ocr_elapsed: float | None = None


class ConvertProgressEstimator:
    def __init__(
        self,
        *,
        input_sup: Path,
        enable_furigana_mask: bool,
        ocr_mode: str,
    ) -> None:
        self._input_sup = input_sup
        self._input_size_bytes = input_sup.stat().st_size if input_sup.exists() else 0
        self._enable_furigana_mask = enable_furigana_mask
        self._ocr_mode = ocr_mode.strip().lower()
        self._clock_base: float | None = None
        self._last_elapsed_sec = 0.0
        self._phase = "idle"
        self._total_rows: int | None = None
        self._wide_total_rows: int | None = None
        self._tall_total_rows: int | None = None
        self._wide_total_unique: int | None = None
        self._tall_total_unique: int | None = None
        self._output_rows: int | None = None
        self._prepare_completed_elapsed: float | None = None
        self._write_started_elapsed: float | None = None
        self._finished = False
        self._active_branch_label: str | None = None
        self._branches: dict[str, _BranchProgressState] = {}

    def record(self, event: ConvertProgressEvent) -> None:
        now = time.monotonic()
        if self._clock_base is None:
            self._clock_base = now - event.elapsed_sec
        self._last_elapsed_sec = max(self._last_elapsed_sec, event.elapsed_sec)
        self._phase = event.phase
        if event.total_rows is not None:
            self._total_rows = event.total_rows
        if event.wide_total_rows is not None:
            self._wide_total_rows = event.wide_total_rows
        if event.tall_total_rows is not None:
            self._tall_total_rows = event.tall_total_rows
        if event.wide_total_unique is not None:
            self._wide_total_unique = event.wide_total_unique
        if event.tall_total_unique is not None:
            self._tall_total_unique = event.tall_total_unique
        if event.output_rows is not None:
            self._output_rows = event.output_rows
        if event.phase == "prepare_completed":
            self._prepare_completed_elapsed = event.elapsed_sec
        if event.phase == "write_started":
            self._write_started_elapsed = event.elapsed_sec
        if event.phase == "finished":
            self._finished = True
        if self._ocr_mode == "fast":
            if self._wide_total_rows is not None or self._wide_total_unique is not None:
                wide_branch = self._branches.setdefault("non-tall-fast", _BranchProgressState())
                if self._wide_total_rows is not None:
                    wide_branch.total_rows = self._wide_total_rows
                if self._wide_total_unique is not None:
                    wide_branch.total_unique = self._wide_total_unique
            if self._tall_total_rows is not None or self._tall_total_unique is not None:
                tall_branch = self._branches.setdefault("tall-default", _BranchProgressState())
                if self._tall_total_rows is not None:
                    tall_branch.total_rows = self._tall_total_rows
                if self._tall_total_unique is not None:
                    tall_branch.total_unique = self._tall_total_unique
        if event.branch_label is None:
            return

        branch = self._branches.setdefault(event.branch_label, _BranchProgressState())
        self._active_branch_label = event.branch_label
        if event.branch_total_rows is not None:
            branch.total_rows = event.branch_total_rows
        if event.branch_total_unique is not None:
            branch.total_unique = event.branch_total_unique
        if event.phase == "backend_loading":
            branch.backend_load_started_elapsed = event.elapsed_sec
        elif event.phase == "backend_ready":
            if branch.backend_load_started_elapsed is not None:
                branch.backend_load_elapsed = max(
                    0.0,
                    event.elapsed_sec - branch.backend_load_started_elapsed,
                )
        elif event.phase == "ocr_started":
            branch.ocr_started_elapsed = event.elapsed_sec
            branch.last_ocr_elapsed = event.elapsed_sec
        elif event.phase == "ocr_progress":
            if event.branch_processed_rows is not None:
                branch.processed_rows = event.branch_processed_rows
            if event.branch_processed_unique is not None:
                branch.processed_unique = event.branch_processed_unique
            branch.last_ocr_elapsed = event.elapsed_sec

    def snapshot(self, *, now_monotonic: float | None = None) -> ConvertProgressSnapshot:
        elapsed_sec = self._current_elapsed_sec(now_monotonic=now_monotonic)
        remaining_sec = self._estimate_remaining_sec(elapsed_sec)
        if self._finished:
            fraction = 1.0
            remaining_sec = 0.0
        elif elapsed_sec <= 0:
            fraction = 0.0
        else:
            fraction = min(0.99, max(0.01, elapsed_sec / max(elapsed_sec + remaining_sec, 0.001)))
        return ConvertProgressSnapshot(
            phase=self._phase,
            headline=self._headline(),
            detail=self._detail(),
            fraction=fraction,
            elapsed_sec=elapsed_sec,
            eta_sec=remaining_sec,
        )

    def _current_elapsed_sec(self, *, now_monotonic: float | None = None) -> float:
        if self._clock_base is None:
            return self._last_elapsed_sec
        if self._finished:
            return self._last_elapsed_sec
        current = (now_monotonic if now_monotonic is not None else time.monotonic()) - self._clock_base
        return max(self._last_elapsed_sec, current)

    def _headline(self) -> str:
        if self._finished:
            return "Done"
        if self._phase.startswith("prepare") or self._phase == "partition_completed":
            return "Prep"
        if self._phase.startswith("backend"):
            return "Load"
        if self._phase.startswith("ocr"):
            return "OCR"
        if self._phase.startswith("write"):
            return "Write"
        return "Run"

    def _detail(self) -> str:
        if self._phase.startswith("ocr") or self._phase.startswith("backend") or self._phase.startswith("write"):
            processed_rows = sum(branch.processed_rows for branch in self._branches.values())
            total_rows = self._known_total_rows()
            if total_rows > 0:
                return f"{processed_rows}/{total_rows}"
        if self._phase == "prepare_completed" and self._total_rows is not None:
            return f"{self._total_rows}"
        if self._finished and self._output_rows is not None:
            return f"{self._output_rows}"
        return ""

    def _known_total_rows(self) -> int:
        if self._total_rows is not None:
            return self._total_rows
        wide_rows, tall_rows = self._estimated_branch_rows()
        return wide_rows + tall_rows

    def _estimated_total_rows(self) -> int:
        if self._total_rows is not None:
            return self._total_rows
        if self._input_size_bytes <= 0:
            return 120
        return max(24, min(6000, int(round(self._input_size_bytes / _ESTIMATED_BYTES_PER_ROW))))

    def _estimated_branch_rows(self) -> tuple[int, int]:
        if self._wide_total_rows is not None or self._tall_total_rows is not None:
            return self._wide_total_rows or 0, self._tall_total_rows or 0
        total_rows = self._estimated_total_rows()
        if self._ocr_mode != "fast":
            return total_rows, 0
        tall_rows = max(0, round(total_rows * 0.12))
        return max(0, total_rows - tall_rows), tall_rows

    def _prepare_prior_sec(self) -> float:
        total_rows = self._estimated_total_rows()
        per_row = 0.0068 if self._enable_furigana_mask else 0.0058
        return max(1.0, 1.0 + total_rows * per_row)

    def _backend_load_prior_sec(self) -> float:
        observed = [
            branch.backend_load_elapsed
            for branch in self._branches.values()
            if branch.backend_load_elapsed is not None
        ]
        if observed:
            return max(2.0, sum(observed) / len(observed))
        return _DEFAULT_BACKEND_LOAD_SEC

    def _write_prior_sec(self) -> float:
        return max(0.2, 0.35 + self._known_total_rows() * 0.0012)

    def _observed_branch_rate_sec(self, branch_label: str) -> float | None:
        branch = self._branches.get(branch_label)
        if branch is None:
            return None
        if branch.processed_unique <= 0:
            return None
        if branch.ocr_started_elapsed is None or branch.last_ocr_elapsed is None:
            return None
        observed_sec = max(0.001, branch.last_ocr_elapsed - branch.ocr_started_elapsed)
        return observed_sec / branch.processed_unique

    def _predicted_tall_rate_from_fast_sec(self) -> float | None:
        if self._ocr_mode != "fast":
            return None
        wide_branch = self._branches.get("non-tall-fast")
        if wide_branch is None:
            return None
        observed_wide_rate = self._observed_branch_rate_sec("non-tall-fast")
        if observed_wide_rate is None:
            return None
        blend = min(wide_branch.processed_unique, _CPU_FAST_TO_TALL_BLEND_UNIQUES) / _CPU_FAST_TO_TALL_BLEND_UNIQUES
        scaled_tall_rate = observed_wide_rate * _CPU_FAST_TO_TALL_RATE_MULTIPLIER
        return (_DEFAULT_TALL_ROW_SEC * (1.0 - blend)) + (scaled_tall_rate * blend)

    def _branch_rate_prior_sec(self, branch_label: str) -> float:
        if branch_label == "non-tall-fast":
            return _DEFAULT_FAST_ROW_SEC
        if branch_label == "tall-default":
            predicted_tall_rate = self._predicted_tall_rate_from_fast_sec()
            if predicted_tall_rate is not None:
                return predicted_tall_rate
            return _DEFAULT_TALL_ROW_SEC
        return _DEFAULT_DEFAULT_ROW_SEC

    def _branch_names(self) -> list[str]:
        if self._ocr_mode != "fast":
            return ["default"]
        names: list[str] = []
        wide_rows, tall_rows = self._estimated_branch_rows()
        if wide_rows > 0:
            names.append("non-tall-fast")
        if tall_rows > 0:
            names.append("tall-default")
        return names

    def _branch_state(self, branch_label: str) -> _BranchProgressState:
        branch = self._branches.get(branch_label)
        if branch is not None:
            return branch
        wide_rows, tall_rows = self._estimated_branch_rows()
        default_rows = self._estimated_total_rows()
        estimated_unique = {
            "non-tall-fast": self._wide_total_unique,
            "tall-default": self._tall_total_unique,
            "default": None,
        }.get(branch_label)
        estimated_rows = {
            "non-tall-fast": wide_rows,
            "tall-default": tall_rows,
            "default": default_rows,
        }.get(branch_label, 0)
        return _BranchProgressState(total_rows=estimated_rows, total_unique=estimated_unique)

    def _estimate_branch_remaining_sec(self, branch_label: str) -> float:
        branch = self._branch_state(branch_label)
        total_unique = branch.total_unique if branch.total_unique is not None else branch.total_rows
        processed_unique = branch.processed_unique
        remaining_unique = max(total_unique - processed_unique, 0)
        prior_rate = self._branch_rate_prior_sec(branch_label)
        observed_rate = self._observed_branch_rate_sec(branch_label)
        if observed_rate is not None:
            blend = min(branch.processed_unique, 6) / 6.0
            rate = (prior_rate * (1.0 - blend)) + (observed_rate * blend)
        else:
            rate = prior_rate
        return remaining_unique * rate

    def _estimate_remaining_sec(self, elapsed_sec: float) -> float:
        if self._finished:
            return 0.0

        prepare_done = self._prepare_completed_elapsed is not None
        if not prepare_done:
            backend_count = len(self._branch_names())
            wide_rows, tall_rows = self._estimated_branch_rows()
            ocr_future = 0.0
            if self._ocr_mode == "fast":
                ocr_future += wide_rows * self._branch_rate_prior_sec("non-tall-fast")
                ocr_future += tall_rows * self._branch_rate_prior_sec("tall-default")
            else:
                ocr_future += (wide_rows + tall_rows) * self._branch_rate_prior_sec("default")
            prepare_remaining = max(self._prepare_prior_sec() - elapsed_sec, 0.4)
            return prepare_remaining + (backend_count * self._backend_load_prior_sec()) + ocr_future + self._write_prior_sec()

        remaining_sec = 0.0
        backend_prior = self._backend_load_prior_sec()
        for branch_label in self._branch_names():
            branch = self._branch_state(branch_label)
            if branch.backend_load_elapsed is None:
                if branch.backend_load_started_elapsed is not None:
                    load_elapsed = max(0.0, elapsed_sec - branch.backend_load_started_elapsed)
                    remaining_sec += max(backend_prior - load_elapsed, 0.25)
                else:
                    remaining_sec += backend_prior
            remaining_sec += self._estimate_branch_remaining_sec(branch_label)

        if self._write_started_elapsed is None:
            remaining_sec += self._write_prior_sec()
        else:
            write_elapsed = max(0.0, elapsed_sec - self._write_started_elapsed)
            remaining_sec += max(self._write_prior_sec() - write_elapsed, 0.05)
        return remaining_sec


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
    progress_callback: Callable[[ConvertProgressEvent], None] | None = None,
    cancel_event: threading.Event | None = None,
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
        progress_callback=progress_callback,
        cancel_event=cancel_event,
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
    if mode is CorrectorMode.GEMINI:
        try:
            require_gemini_api_key(request.corrector_api_key_env)
        except GeminiAuthError as exc:
            raise ConvertPreparationError(str(exc)) from exc

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
