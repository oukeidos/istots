from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from istots import model_store


class SetupArgumentError(ValueError):
    pass


class SetupExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SetupProgressEvent:
    phase: str
    headline: str
    detail: str
    fraction: float | None = None


@dataclass(frozen=True)
class SetupRequest:
    with_hf_fallback: bool = False
    model_id: str = model_store.DEFAULT_MODEL_ID
    gguf_model_id: str = model_store.DEFAULT_GGUF_MODEL_ID
    with_qwen_corrector: bool = False
    qwen_corrector_model_id: str = model_store.DEFAULT_QWEN_CORRECTOR_MODEL_ID
    qwen_corrector_model_filename: str = model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME
    qwen_corrector_mmproj_filename: str = model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME
    models_dir: Path | None = None
    force: bool = False
    support_dir: Path | None = None
    gguf_py_base_url: str | None = None
    gguf_source_mode: str = "auto"
    min_pixels: int = 32768
    derived_mmproj_output_path: Path | None = None
    bootstrap_managed_runtime: bool = False
    runtime_variant: str = "auto"
    install_prerequisites: bool = False


@dataclass(frozen=True)
class SetupResult:
    artifacts: model_store.SetupArtifacts
    custom_hf_bundle: bool
    custom_gguf_bundle: bool
    custom_qwen_bundle: bool


def execute_setup_request(
    request: SetupRequest,
    *,
    progress_callback: Callable[[SetupProgressEvent], None] | None = None,
    emit_completion_event: bool = True,
) -> SetupResult:
    if not request.with_hf_fallback and not model_store.is_default_pinned_hf_model(request.model_id):
        raise SetupArgumentError("--model-id requires --with-hf-fallback")

    try:
        if request.bootstrap_managed_runtime:
            from istots.gui.bootstrap_windows import install_managed_llama_cpp_runtime

            install_managed_llama_cpp_runtime(
                requested_variant=request.runtime_variant,
                force=request.force,
                install_prerequisites=request.install_prerequisites,
                progress_callback=lambda phase, headline, detail, fraction: _emit_progress(
                    progress_callback,
                    phase=phase,
                    headline=headline,
                    detail=detail,
                    fraction=fraction,
                ),
            )
        _emit_progress(
            progress_callback,
            phase="model_setup",
            headline="Setup Assets",
            detail="Provisioning model and mmproj assets",
            fraction=0.80 if request.bootstrap_managed_runtime else 0.20,
        )
        artifacts = model_store.setup_default_runtime_assets(
            hf_model_id=request.model_id,
            gguf_model_id=request.gguf_model_id,
            with_hf_fallback=request.with_hf_fallback,
            with_qwen_corrector=request.with_qwen_corrector,
            qwen_corrector_model_id=request.qwen_corrector_model_id,
            qwen_corrector_model_filename=request.qwen_corrector_model_filename,
            qwen_corrector_mmproj_filename=request.qwen_corrector_mmproj_filename,
            models_dir=request.models_dir,
            force=request.force,
            support_dir=request.support_dir,
            gguf_py_base_url=request.gguf_py_base_url,
            gguf_source_mode=request.gguf_source_mode,
            min_pixels=request.min_pixels,
            derived_mmproj_output_path=request.derived_mmproj_output_path,
        )
    except Exception as exc:
        raise SetupExecutionError(str(exc)) from exc

    if emit_completion_event:
        _emit_progress(
            progress_callback,
            phase="complete",
            headline="Setup Complete",
            detail="Runtime assets are ready",
            fraction=1.0,
        )

    return SetupResult(
        artifacts=artifacts,
        custom_hf_bundle=request.with_hf_fallback and not model_store.is_default_pinned_hf_model(request.model_id),
        custom_gguf_bundle=not model_store.is_default_pinned_gguf_model(request.gguf_model_id),
        custom_qwen_bundle=request.with_qwen_corrector
        and not model_store.is_default_pinned_qwen_bundle(
            model_id=request.qwen_corrector_model_id,
            model_filename=request.qwen_corrector_model_filename,
            mmproj_filename=request.qwen_corrector_mmproj_filename,
        ),
    )


def _emit_progress(
    callback: Callable[[SetupProgressEvent], None] | None,
    *,
    phase: str,
    headline: str,
    detail: str,
    fraction: float | None,
) -> None:
    if callback is None:
        return
    callback(
        SetupProgressEvent(
            phase=phase,
            headline=headline,
            detail=detail,
            fraction=fraction,
        )
    )
