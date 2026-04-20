from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from istots import model_store


class SetupArgumentError(ValueError):
    pass


class SetupExecutionError(RuntimeError):
    pass


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


@dataclass(frozen=True)
class SetupResult:
    artifacts: model_store.SetupArtifacts
    custom_hf_bundle: bool
    custom_gguf_bundle: bool
    custom_qwen_bundle: bool


def execute_setup_request(request: SetupRequest) -> SetupResult:
    if not request.with_hf_fallback and not model_store.is_default_pinned_hf_model(request.model_id):
        raise SetupArgumentError("--model-id requires --with-hf-fallback")

    try:
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
        )
    except Exception as exc:
        raise SetupExecutionError(str(exc)) from exc

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
