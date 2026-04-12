"""OCR backend contracts and factories."""

from .factory import create_ocr_backend
from .types import (
    OCRBackend,
    OCRBackendConfig,
    OCREngine,
    PaddleOCRVLRuntimeOverrides,
    Qwen35RuntimeOverrides,
    ResolvedLlamaRuntimeOverrides,
    normalize_ocr_engine,
)

__all__ = [
    "OCRBackend",
    "OCRBackendConfig",
    "OCREngine",
    "PaddleOCRVLRuntimeOverrides",
    "Qwen35RuntimeOverrides",
    "ResolvedLlamaRuntimeOverrides",
    "create_ocr_backend",
    "normalize_ocr_engine",
]
