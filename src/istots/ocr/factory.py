from __future__ import annotations

from .types import OCRBackend, OCRBackendConfig, OCREngine, normalize_ocr_engine


def create_ocr_backend(config: OCRBackendConfig) -> OCRBackend:
    engine = normalize_ocr_engine(config.engine)
    normalized = OCRBackendConfig(
        engine=engine,
        model_id=config.model_id,
        device=config.device,
        max_new_tokens=config.max_new_tokens,
        local_files_only=config.local_files_only,
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
    raise NotImplementedError(
        "The llama-server OCR backend is not implemented yet. "
        "Step 1 only introduces the engine boundary."
    )
