from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence, runtime_checkable

from PIL import Image


class OCREngine(StrEnum):
    HF = "hf"
    LLAMA_SERVER = "llama-server"


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
    device: str = "cpu"
    max_new_tokens: int = 256
    local_files_only: bool = True


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
