from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from istots import doctor as doctor_module
from istots.ocr import PaddleOCRVLRuntimeOverrides, Qwen35RuntimeOverrides


class DoctorArgumentError(ValueError):
    pass


@dataclass(frozen=True)
class DoctorIssue:
    code: str
    message: str


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    issues: tuple[DoctorIssue, ...]
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DoctorResult:
    category: str
    target: str
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


@dataclass(frozen=True)
class DoctorRequest:
    category: str | None = None
    target: str | None = None
    models_dir: Path | None = None
    min_pixels: int = 32768
    explicit_binary_path: Path | None = None
    host: str = "127.0.0.1"
    input_sup: Path | None = None
    api_key_env: str = "GEMINI_API_KEY"
    paddle_profile: str = "auto"
    paddle_port: int | None = None
    paddle_threads: int | None = None
    paddle_threads_batch: int | None = None
    paddle_gpu_layers: int | None = None
    paddle_no_mmproj_offload: bool = False
    paddle_startup_timeout_sec: float = 120.0
    paddle_ctx_size: int | None = None
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
    use_temp_ocr_image_files: bool = True


@dataclass(frozen=True)
class DoctorExecutionPlan:
    category: str
    target: str
    models_dir: Path | None
    min_pixels: int
    explicit_binary_path: Path | None
    host: str
    input_sup: Path | None
    api_key_env: str
    paddle_overrides: PaddleOCRVLRuntimeOverrides
    qwen_overrides: Qwen35RuntimeOverrides
    corrector_model_path: Path | None
    corrector_mmproj_path: Path | None
    paddle_startup_timeout_sec: float
    qwen_startup_timeout_sec: float
    use_temp_ocr_image_files: bool


def plan_doctor_request(request: DoctorRequest) -> DoctorExecutionPlan:
    category = request.category
    target = request.target
    if category is None:
        raise DoctorArgumentError("doctor requires a category: runtime, auth, or workflow")
    if target is None:
        raise DoctorArgumentError("doctor category requires a target")

    normalized_target = str(target).strip().lower()
    allowed_targets = {
        "runtime": {"paddle", "qwen"},
        "auth": {"gemini"},
        "workflow": {"default", "wider", "corrector-qwen", "corrector-gemini"},
    }
    if normalized_target not in allowed_targets[category]:
        joined = ", ".join(sorted(allowed_targets[category]))
        raise DoctorArgumentError(
            f"unsupported doctor target for {category}: {target!r}. Expected one of: {joined}"
        )
    if category == "workflow" and request.input_sup is None:
        raise DoctorArgumentError("doctor workflow requires --input-sup")

    return DoctorExecutionPlan(
        category=category,
        target=normalized_target,
        models_dir=request.models_dir,
        min_pixels=request.min_pixels,
        explicit_binary_path=request.explicit_binary_path,
        host=request.host,
        input_sup=request.input_sup.expanduser().resolve() if request.input_sup is not None else None,
        api_key_env=request.api_key_env,
        paddle_overrides=PaddleOCRVLRuntimeOverrides(
            profile=request.paddle_profile,
            port=request.paddle_port,
            threads=request.paddle_threads,
            threads_batch=request.paddle_threads_batch,
            gpu_layers=request.paddle_gpu_layers,
            no_mmproj_offload=True if request.paddle_no_mmproj_offload else None,
            startup_timeout_sec=request.paddle_startup_timeout_sec,
            ctx_size=request.paddle_ctx_size,
        ),
        qwen_overrides=Qwen35RuntimeOverrides(
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
        corrector_model_path=request.corrector_model_path,
        corrector_mmproj_path=request.corrector_mmproj_path,
        paddle_startup_timeout_sec=request.paddle_startup_timeout_sec,
        qwen_startup_timeout_sec=request.qwen_startup_timeout_sec,
        use_temp_ocr_image_files=request.use_temp_ocr_image_files,
    )


def execute_doctor_plan(plan: DoctorExecutionPlan) -> DoctorResult:
    raw_result = None
    if plan.category == "runtime" and plan.target == "paddle":
        raw_result = doctor_module.run_paddle_runtime_doctor(
            models_dir=plan.models_dir,
            min_pixels=plan.min_pixels,
            explicit_binary_path=plan.explicit_binary_path,
            host=plan.host,
            overrides=plan.paddle_overrides,
            startup_timeout_sec=plan.paddle_startup_timeout_sec,
        )
    elif plan.category == "runtime" and plan.target == "qwen":
        raw_result = doctor_module.run_qwen_runtime_doctor(
            models_dir=plan.models_dir,
            explicit_binary_path=plan.explicit_binary_path,
            host=plan.host,
            overrides=plan.qwen_overrides,
            explicit_model_path=plan.corrector_model_path,
            explicit_mmproj_path=plan.corrector_mmproj_path,
            startup_timeout_sec=plan.qwen_startup_timeout_sec,
        )
    elif plan.category == "auth" and plan.target == "gemini":
        raw_result = doctor_module.run_gemini_auth_doctor(api_key_env=plan.api_key_env)
    elif plan.category == "workflow":
        raw_result = doctor_module.run_workflow_doctor(
            workflow=plan.target,
            input_sup=plan.input_sup,
            models_dir=plan.models_dir,
            min_pixels=plan.min_pixels,
            explicit_binary_path=plan.explicit_binary_path,
            host=plan.host,
            paddle_overrides=plan.paddle_overrides,
            qwen_overrides=plan.qwen_overrides,
            explicit_qwen_model_path=plan.corrector_model_path,
            explicit_qwen_mmproj_path=plan.corrector_mmproj_path,
            api_key_env=plan.api_key_env,
            startup_timeout_sec=max(plan.paddle_startup_timeout_sec, plan.qwen_startup_timeout_sec),
            use_temp_ocr_image_files=plan.use_temp_ocr_image_files,
        )
    else:
        raise DoctorArgumentError("unsupported doctor mode")
    return DoctorResult(
        category=raw_result.category,
        target=raw_result.target,
        checks=tuple(
            DoctorCheck(
                name=check.name,
                ok=check.ok,
                issues=tuple(
                    DoctorIssue(code=issue.code, message=issue.message)
                    for issue in check.issues
                ),
                details=tuple(check.details),
            )
            for check in raw_result.checks
        ),
    )
