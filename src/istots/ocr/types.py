from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from PIL import Image


class OCREngine(StrEnum):
    HF = "hf"
    LLAMA_SERVER = "llama-server"


LOCAL_PADDLE_CTX_SIZE = 2048
LOCAL_PADDLE_MAX_REQUESTS_PER_INSTANCE = 200


def normalize_ocr_engine(engine: str | OCREngine) -> OCREngine:
    if isinstance(engine, OCREngine):
        return engine

    try:
        return OCREngine(engine)
    except ValueError as exc:
        supported = ", ".join(item.value for item in OCREngine)
        raise ValueError(
            f"unsupported OCR engine: {engine!r}. Expected one of: {supported}"
        ) from exc


@dataclass(frozen=True)
class OCRBackendConfig:
    engine: OCREngine = OCREngine.HF
    model_id: str = ""
    model_path: Path | None = None
    mmproj_path: Path | None = None
    device: str | None = None
    hf_dtype: str = "auto"
    hf_min_pixels: int | None = None
    max_new_tokens: int = 256
    local_files_only: bool = True
    models_dir: Path | None = None
    role: str = "ocr"
    prompt_text: str = "OCR:"
    profile: str = "auto"
    binary_path: Path | None = None
    host: str = "127.0.0.1"
    port: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    ctx_size: int | None = None
    n_predict: int | None = None
    reasoning: str | None = None
    reasoning_budget: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool | None = None
    startup_timeout_sec: float = 120.0
    max_requests_per_instance: int | None = None


@dataclass(frozen=True)
class PaddleOCRVLRuntimeOverrides:
    profile: str = "auto"
    port: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool | None = None
    startup_timeout_sec: float = 120.0
    ctx_size: int | None = None


@dataclass(frozen=True)
class Qwen35RuntimeOverrides:
    profile: str = "auto"
    port: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool | None = None
    startup_timeout_sec: float = 120.0
    ctx_size: int | None = None
    n_predict: int | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class ResolvedLlamaRuntimeOverrides:
    profile: str = "auto"
    port: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool | None = None
    startup_timeout_sec: float = 120.0
    ctx_size: int | None = None
    n_predict: int | None = None
    reasoning: str | None = None


@runtime_checkable
class OCRBackend(Protocol):
    def recognize(self, image: Image.Image) -> str:
        ...

    def recognize_batch(self, images: Sequence[Image.Image]) -> list[str]:
        ...

    def clear_device_cache(self) -> None:
        ...

    def close(self) -> None:
        ...


def resolve_llama_server_request_budget(explicit: int | None, *, default: int | None = None) -> int | None:
    if explicit is not None:
        return explicit if explicit > 0 else None
    raw = os.environ.get("ISTOTS_LLAMA_SERVER_MAX_REQUESTS_PER_INSTANCE")
    if raw is not None:
        value = int(raw)
        return value if value > 0 else None
    if default is None:
        return None
    return default if default > 0 else None
