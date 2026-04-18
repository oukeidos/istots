from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from istots.corrector import (
    LOCAL_QWEN_CTX_SIZE,
    LOCAL_QWEN_MAX_NEW_TOKENS,
    CorrectorConfig,
    CorrectorMode,
    STRICT_OCR_V1_PROMPT,
)
from istots.gemini_auth import GeminiAuthStatus, get_gemini_auth_status
from istots.llama_runtime import (
    DEFAULT_LLAMA_SERVER_HOST,
    DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SEC,
    DEFAULT_ROLE_PORTS,
    LlamaServerDoctorIssue,
    LlamaServerDoctorReport,
    LlamaServerLaunchSpec,
    LlamaServerProfile,
    LlamaServerRole,
    detect_llama_server_path,
    run_llama_server_doctor,
    run_llama_server_launch_spec_doctor,
)
from istots.model_store import ensure_local_qwen_corrector_assets
from istots.ocr import (
    LOCAL_PADDLE_CTX_SIZE,
    OCREngine,
    PaddleOCRVLRuntimeOverrides,
    Qwen35RuntimeOverrides,
)
from istots.pipeline import convert_sup_to_srt


@dataclass(frozen=True)
class DoctorCheckResult:
    name: str
    ok: bool
    issues: tuple[LlamaServerDoctorIssue, ...] = ()
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DoctorSuiteResult:
    category: str
    target: str
    checks: tuple[DoctorCheckResult, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _report_to_check(name: str, report: LlamaServerDoctorReport) -> DoctorCheckResult:
    def _detail_value(value: object) -> str:
        return str(getattr(value, "value", value))

    details: list[tuple[str, str]] = [
        ("role", _detail_value(report.role)),
        ("profile", _detail_value(report.profile)),
    ]
    if report.launch_spec is not None:
        details.extend(
            [
                ("binary", str(report.launch_spec.binary_path)),
                ("model", str(report.launch_spec.model_path)),
                ("mmproj", str(report.launch_spec.mmproj_path)),
                ("host", report.launch_spec.host),
                ("port", str(report.launch_spec.port)),
            ]
        )
    if report.smoke_response is not None:
        details.append(("smoke_response", report.smoke_response))
    return DoctorCheckResult(
        name=name,
        ok=report.ok,
        issues=report.issues,
        details=tuple(details),
    )


def run_paddle_runtime_doctor(
    *,
    models_dir: Path | None = None,
    min_pixels: int = 32768,
    explicit_binary_path: Path | None = None,
    host: str = DEFAULT_LLAMA_SERVER_HOST,
    overrides: PaddleOCRVLRuntimeOverrides | None = None,
    startup_timeout_sec: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SEC,
) -> DoctorSuiteResult:
    normalized_overrides = overrides or PaddleOCRVLRuntimeOverrides()
    if normalized_overrides.ctx_size is None:
        normalized_overrides = replace(normalized_overrides, ctx_size=LOCAL_PADDLE_CTX_SIZE)
    checks = [
        _report_to_check(
            f"runtime:{role.value}",
            run_llama_server_doctor(
                role=role,
                models_dir=models_dir,
                min_pixels=min_pixels,
                explicit_binary_path=explicit_binary_path,
                host=host,
                overrides=normalized_overrides,
                startup_timeout_sec=startup_timeout_sec,
            ),
        )
        for role in (LlamaServerRole.OCR, LlamaServerRole.OCR_FAST, LlamaServerRole.DETECTOR)
    ]
    return DoctorSuiteResult(category="runtime", target="paddle", checks=tuple(checks))


def _resolve_qwen_runtime_overrides(
    overrides: Qwen35RuntimeOverrides | None,
) -> Qwen35RuntimeOverrides:
    normalized = overrides or Qwen35RuntimeOverrides()
    return Qwen35RuntimeOverrides(
        profile=normalized.profile,
        port=normalized.port or DEFAULT_ROLE_PORTS[LlamaServerRole.CORRECTOR],
        threads=normalized.threads,
        threads_batch=normalized.threads_batch,
        gpu_layers=normalized.gpu_layers,
        no_mmproj_offload=normalized.no_mmproj_offload,
        startup_timeout_sec=normalized.startup_timeout_sec,
        ctx_size=normalized.ctx_size if normalized.ctx_size is not None else LOCAL_QWEN_CTX_SIZE,
        n_predict=normalized.n_predict if normalized.n_predict is not None else LOCAL_QWEN_MAX_NEW_TOKENS,
        reasoning=normalized.reasoning if normalized.reasoning is not None else "off",
    )


def run_qwen_runtime_doctor(
    *,
    models_dir: Path | None = None,
    explicit_binary_path: Path | None = None,
    host: str = DEFAULT_LLAMA_SERVER_HOST,
    overrides: Qwen35RuntimeOverrides | None = None,
    explicit_model_path: Path | None = None,
    explicit_mmproj_path: Path | None = None,
    startup_timeout_sec: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SEC,
) -> DoctorSuiteResult:
    binary_path = detect_llama_server_path(explicit_binary_path)
    if binary_path is None:
        return DoctorSuiteResult(
            category="runtime",
            target="qwen",
            checks=(
                DoctorCheckResult(
                    name="runtime:qwen",
                    ok=False,
                    issues=(
                        LlamaServerDoctorIssue(
                            code="missing_binary",
                            message="llama-server binary not found. Set ISTOTS_LLAMA_SERVER_PATH or pass --llama-server-path.",
                        ),
                    ),
                ),
            ),
        )

    try:
        if explicit_model_path is not None or explicit_mmproj_path is not None:
            if explicit_model_path is None or explicit_mmproj_path is None:
                raise RuntimeError(
                    "Qwen runtime doctor requires both --corrector-model-path and --corrector-mmproj-path when either is provided."
                )
            model_path = explicit_model_path.expanduser().resolve()
            mmproj_path = explicit_mmproj_path.expanduser().resolve()
        else:
            model_path, mmproj_path = ensure_local_qwen_corrector_assets(models_dir=models_dir)
    except Exception as exc:
        return DoctorSuiteResult(
            category="runtime",
            target="qwen",
            checks=(
                DoctorCheckResult(
                    name="runtime:qwen",
                    ok=False,
                    issues=(LlamaServerDoctorIssue(code="asset_resolution_failed", message=str(exc)),),
                ),
            ),
        )

    resolved = _resolve_qwen_runtime_overrides(overrides)
    profile = LlamaServerProfile(resolved.profile)
    spec = LlamaServerLaunchSpec(
        role=LlamaServerRole.CORRECTOR,
        profile=profile,
        binary_path=binary_path,
        model_path=model_path,
        mmproj_path=mmproj_path,
        host=host,
        port=int(resolved.port or DEFAULT_ROLE_PORTS[LlamaServerRole.CORRECTOR]),
        threads=resolved.threads,
        threads_batch=resolved.threads_batch,
        ctx_size=resolved.ctx_size,
        n_predict=resolved.n_predict,
        reasoning=resolved.reasoning,
        gpu_layers=resolved.gpu_layers,
        no_mmproj_offload=bool(resolved.no_mmproj_offload),
        prompt_text=STRICT_OCR_V1_PROMPT,
    )
    report = run_llama_server_launch_spec_doctor(
        spec,
        startup_timeout_sec=startup_timeout_sec,
    )
    return DoctorSuiteResult(
        category="runtime",
        target="qwen",
        checks=(_report_to_check("runtime:qwen", report),),
    )


def run_gemini_auth_doctor(
    *,
    api_key_env: str = "GEMINI_API_KEY",
) -> DoctorSuiteResult:
    status = get_gemini_auth_status(api_key_env)
    issues: list[LlamaServerDoctorIssue] = []
    if status.effective_source is None:
        issues.append(
            LlamaServerDoctorIssue(
                code="missing_credentials",
                message=(
                    "no usable Gemini API key source found. "
                    "Run `istots auth gemini set`, configure `istots auth gemini env-file set PATH`, "
                    f"or export {api_key_env}."
                ),
            )
        )
    check = DoctorCheckResult(
        name="auth:gemini",
        ok=not issues,
        issues=tuple(issues),
        details=_gemini_status_details(status),
    )
    return DoctorSuiteResult(category="auth", target="gemini", checks=(check,))


def _gemini_status_details(status: GeminiAuthStatus) -> tuple[tuple[str, str], ...]:
    details = [
        ("keyring_backend", status.keyring_backend or "missing"),
        ("keyring_configured", "yes" if status.keyring_configured else "no"),
        ("env_file", str(status.env_file_path) if status.env_file_path is not None else "unset"),
        ("env_file_contains_key", "yes" if status.env_file_contains_key else "no"),
        ("shell_env", status.process_env_name or "missing"),
        ("effective_source", status.effective_source or "missing"),
    ]
    return tuple(details)


def run_workflow_doctor(
    *,
    workflow: str,
    input_sup: Path,
    models_dir: Path | None = None,
    min_pixels: int = 32768,
    explicit_binary_path: Path | None = None,
    host: str = DEFAULT_LLAMA_SERVER_HOST,
    paddle_overrides: PaddleOCRVLRuntimeOverrides | None = None,
    qwen_overrides: Qwen35RuntimeOverrides | None = None,
    explicit_qwen_model_path: Path | None = None,
    explicit_qwen_mmproj_path: Path | None = None,
    api_key_env: str = "GEMINI_API_KEY",
    startup_timeout_sec: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SEC,
) -> DoctorSuiteResult:
    normalized = workflow.strip().lower()
    checks: list[DoctorCheckResult] = []
    checks.append(
        DoctorCheckResult(
            name="workflow:input-sup",
            ok=input_sup.exists(),
            issues=tuple()
            if input_sup.exists()
            else (
                LlamaServerDoctorIssue(
                    code="missing_input_sup",
                    message=f"workflow input SUP is missing: {input_sup}",
                ),
            ),
            details=(("input_sup", str(input_sup)),),
        )
    )

    if normalized in {"default", "wider", "corrector-qwen", "corrector-gemini"} and input_sup.exists():
        workflow_corrector: CorrectorConfig | None = None
        if normalized == "corrector-qwen":
            try:
                if explicit_qwen_model_path is not None or explicit_qwen_mmproj_path is not None:
                    if explicit_qwen_model_path is None or explicit_qwen_mmproj_path is None:
                        raise RuntimeError(
                            "workflow corrector-qwen requires both explicit Qwen paths when either path is provided."
                        )
                    model_path = explicit_qwen_model_path.expanduser().resolve()
                    mmproj_path = explicit_qwen_mmproj_path.expanduser().resolve()
                else:
                    model_path, mmproj_path = ensure_local_qwen_corrector_assets(models_dir=models_dir)
                workflow_corrector = CorrectorConfig(
                    mode=CorrectorMode.QWEN_LOCAL,
                    local_model_path=model_path,
                    local_mmproj_path=mmproj_path,
                    local_runtime_overrides=qwen_overrides or Qwen35RuntimeOverrides(),
                )
            except Exception as exc:
                checks.append(
                    DoctorCheckResult(
                        name="workflow:smoke",
                        ok=False,
                        issues=(LlamaServerDoctorIssue(code="corrector_config_failed", message=str(exc)),),
                    )
                )
            qwen_suite = run_qwen_runtime_doctor(
                models_dir=models_dir,
                explicit_binary_path=explicit_binary_path,
                host=host,
                overrides=qwen_overrides,
                explicit_model_path=explicit_qwen_model_path,
                explicit_mmproj_path=explicit_qwen_mmproj_path,
                startup_timeout_sec=startup_timeout_sec,
            )
            checks.extend(qwen_suite.checks)
        elif normalized == "corrector-gemini":
            workflow_corrector = CorrectorConfig(
                mode=CorrectorMode.GEMINI,
                api_key_env=api_key_env,
            )
            auth_suite = run_gemini_auth_doctor(api_key_env=api_key_env)
            checks.extend(auth_suite.checks)

        if not any(check.name == "workflow:smoke" and not check.ok for check in checks):
            checks.append(
                _run_workflow_smoke_check(
                    input_sup=input_sup,
                    models_dir=models_dir,
                    detector_mode="wider" if normalized == "wider" else "default",
                    explicit_binary_path=explicit_binary_path,
                    host=host,
                    min_pixels=min_pixels,
                    paddle_overrides=paddle_overrides or PaddleOCRVLRuntimeOverrides(),
                    corrector_config=workflow_corrector,
                )
            )

    return DoctorSuiteResult(category="workflow", target=normalized, checks=tuple(checks))


def _run_workflow_smoke_check(
    *,
    input_sup: Path,
    models_dir: Path | None,
    detector_mode: str,
    explicit_binary_path: Path | None,
    host: str,
    min_pixels: int,
    paddle_overrides: PaddleOCRVLRuntimeOverrides,
    corrector_config: CorrectorConfig | None,
) -> DoctorCheckResult:
    output_dir = Path(tempfile.mkdtemp(prefix="istots-doctor-workflow-")).resolve()
    output_srt = output_dir / "doctor.srt"
    detector_output = output_dir / "doctor.detector.jsonl"
    corrector_output = output_dir / "doctor.corrected.jsonl"
    if corrector_config is not None:
        corrector_config = replace(corrector_config, output_path=corrector_output)
    try:
        result = convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            engine=OCREngine.LLAMA_SERVER,
            ocr_mode="default",
            detector_output=detector_output,
            detector_mode=detector_mode,
            corrector_config=corrector_config,
            model_id="PaddlePaddle/PaddleOCR-VL-1.5",
            models_dir=models_dir,
            local_files_only=True,
            runtime_binary_path=explicit_binary_path,
            runtime_host=host,
            paddle_runtime_overrides=paddle_overrides,
            verbose=False,
        )
    except Exception as exc:
        return DoctorCheckResult(
            name="workflow:smoke",
            ok=False,
            issues=(LlamaServerDoctorIssue(code="workflow_smoke_failed", message=str(exc)),),
        )

    details = [
        ("output_srt", str(result.output_srt)),
        ("processed_count", str(result.processed_count)),
        ("written_count", str(result.written_count)),
        ("detector_record_count", str(result.detector_record_count)),
    ]
    if corrector_config is not None:
        details.append(("correction_record_count", str(result.correction_record_count)))
        details.append(("correction_applied_count", str(result.correction_applied_count)))
    return DoctorCheckResult(
        name="workflow:smoke",
        ok=True,
        details=tuple(details),
    )
