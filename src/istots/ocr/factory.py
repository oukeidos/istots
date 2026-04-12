from __future__ import annotations

from .types import OCRBackend, OCRBackendConfig, OCREngine, normalize_ocr_engine


def create_ocr_backend(config: OCRBackendConfig) -> OCRBackend:
    engine = normalize_ocr_engine(config.engine)
    normalized = OCRBackendConfig(
        engine=engine,
        model_id=config.model_id,
        model_path=config.model_path,
        mmproj_path=config.mmproj_path,
        device=config.device,
        max_new_tokens=config.max_new_tokens,
        local_files_only=config.local_files_only,
        models_dir=config.models_dir,
        role=config.role,
        prompt_text=config.prompt_text,
        profile=config.profile,
        binary_path=config.binary_path,
        host=config.host,
        port=config.port,
        threads=config.threads,
        threads_batch=config.threads_batch,
        ctx_size=config.ctx_size,
        n_predict=config.n_predict,
        reasoning=config.reasoning,
        reasoning_budget=config.reasoning_budget,
        gpu_layers=config.gpu_layers,
        no_mmproj_offload=config.no_mmproj_offload,
        startup_timeout_sec=config.startup_timeout_sec,
    )
    if engine is OCREngine.HF:
        return _create_hf_backend(normalized)
    if engine is OCREngine.LLAMA_SERVER:
        return _create_llama_server_backend(normalized)
    raise AssertionError(f"unhandled OCR engine: {engine}")


def _create_hf_backend(config: OCRBackendConfig) -> OCRBackend:
    from .hf_backend import HFPaddleOCRVLBackend

    return HFPaddleOCRVLBackend(
        model_id=config.model_id,
        device=config.device,
        max_new_tokens=config.max_new_tokens,
        local_files_only=config.local_files_only,
    )


def _create_llama_server_backend(config: OCRBackendConfig) -> OCRBackend:
    from .llama_server_backend import LlamaServerOCRBackend

    return LlamaServerOCRBackend(
        device=config.device,
        max_new_tokens=config.max_new_tokens,
        model_path=config.model_path,
        mmproj_path=config.mmproj_path,
        models_dir=config.models_dir,
        role=config.role,
        prompt_text=config.prompt_text,
        profile=config.profile,
        binary_path=config.binary_path,
        host=config.host,
        port=config.port,
        threads=config.threads,
        threads_batch=config.threads_batch,
        ctx_size=config.ctx_size,
        n_predict=config.n_predict,
        reasoning=config.reasoning,
        reasoning_budget=config.reasoning_budget,
        gpu_layers=config.gpu_layers,
        no_mmproj_offload=config.no_mmproj_offload,
        startup_timeout_sec=config.startup_timeout_sec,
    )
