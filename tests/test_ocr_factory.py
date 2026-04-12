from __future__ import annotations

import pytest

from istots.ocr import OCRBackendConfig, OCREngine, create_ocr_backend, normalize_ocr_engine
from istots.ocr import factory


def test_normalize_ocr_engine_accepts_known_values() -> None:
    assert normalize_ocr_engine("hf") is OCREngine.HF
    assert normalize_ocr_engine("llama-server") is OCREngine.LLAMA_SERVER
    assert normalize_ocr_engine(OCREngine.HF) is OCREngine.HF


def test_normalize_ocr_engine_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unsupported OCR engine"):
        normalize_ocr_engine("bogus")


def test_create_ocr_backend_routes_hf(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_create_hf_backend(config: OCRBackendConfig):
        captured["config"] = config
        return sentinel

    monkeypatch.setattr(factory, "_create_hf_backend", fake_create_hf_backend)

    result = create_ocr_backend(
        OCRBackendConfig(
            engine=OCREngine.HF,
            model_id="org/model",
            device="cpu",
            max_new_tokens=123,
            local_files_only=False,
        )
    )

    assert result is sentinel
    assert captured["config"] == OCRBackendConfig(
        engine=OCREngine.HF,
        model_id="org/model",
        device="cpu",
        max_new_tokens=123,
        local_files_only=False,
    )


def test_create_ocr_backend_routes_llama_server(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_create_llama_backend(config: OCRBackendConfig):
        captured["config"] = config
        return sentinel

    monkeypatch.setattr(factory, "_create_llama_server_backend", fake_create_llama_backend)

    result = create_ocr_backend(OCRBackendConfig(engine=OCREngine.LLAMA_SERVER, model_id="unused"))

    assert result is sentinel
    assert captured["config"].engine is OCREngine.LLAMA_SERVER
