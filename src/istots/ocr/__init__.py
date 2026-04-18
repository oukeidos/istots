"""OCR backend contracts and factories."""

from .factory import create_ocr_backend
from .types import (
    LOCAL_PADDLE_MAX_REQUESTS_PER_INSTANCE,
    LOCAL_PADDLE_CTX_SIZE,
    OCRBackend,
    OCRBackendConfig,
    OCREngine,
    PaddleOCRVLRuntimeOverrides,
    Qwen35RuntimeOverrides,
    ResolvedLlamaRuntimeOverrides,
    normalize_ocr_engine,
)

__all__ = [
    "LOCAL_PADDLE_MAX_REQUESTS_PER_INSTANCE",
    "LOCAL_PADDLE_CTX_SIZE",
    "OCRBackend",
    "OCRBackendConfig",
    "OCREngine",
    "PaddleOCRVLRuntimeOverrides",
    "Qwen35RuntimeOverrides",
    "ResolvedLlamaRuntimeOverrides",
    "create_ocr_backend",
    "normalize_ocr_engine",
]
