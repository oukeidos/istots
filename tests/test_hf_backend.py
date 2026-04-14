from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from istots.ocr.hf_backend import HFPaddleOCRVLBackend, _HFRecognitionResult


def test_hf_backend_missing_optional_runtime_mentions_extra(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch", "transformers"}:
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="uv sync --extra hf"):
        HFPaddleOCRVLBackend(
            model_id="org/model",
            device="cpu",
        )


def test_hf_backend_applies_min_pixels_override(monkeypatch) -> None:
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="right"),
        image_processor=SimpleNamespace(min_pixels=None),
    )

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def eval(self) -> None:
            return None

    fake_torch = SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        cuda=SimpleNamespace(empty_cache=lambda: None, ipc_collect=lambda: None),
    )
    fake_transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForImageTextToText=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeModel()
        ),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = HFPaddleOCRVLBackend(
        model_id="org/model",
        device="cpu",
        dtype="float32",
        min_pixels_override=32768,
    )

    assert processor.tokenizer.padding_side == "left"
    assert processor.image_processor.min_pixels == 32768
    backend.close()


def test_hf_backend_applies_length_fallback_for_main_ocr() -> None:
    backend = object.__new__(HFPaddleOCRVLBackend)
    backend.role = "ocr"
    backend.prompt_text = "OCR:"
    backend.max_new_tokens = 256

    prompts: list[str] = []

    def fake_recognize_once(image, *, prompt_text: str):
        prompts.append(prompt_text)
        if prompt_text == "OCR:":
            return _HFRecognitionResult(
                normalized_text="bad",
                generated_token_count=256,
                hit_max_new_tokens=True,
            )
        return _HFRecognitionResult(
            normalized_text="good",
            generated_token_count=12,
            hit_max_new_tokens=False,
        )

    backend._recognize_once = fake_recognize_once

    assert backend.recognize(Image.new("RGB", (2, 2), "white")) == "good"
    assert prompts == ["OCR:", "Output only the exact text."]


def test_hf_backend_skips_length_fallback_outside_main_ocr() -> None:
    backend = object.__new__(HFPaddleOCRVLBackend)
    backend.role = "detector"
    backend.prompt_text = "OCR:"
    backend.max_new_tokens = 256

    prompts: list[str] = []

    def fake_recognize_once(image, *, prompt_text: str):
        prompts.append(prompt_text)
        return _HFRecognitionResult(
            normalized_text="kept",
            generated_token_count=256,
            hit_max_new_tokens=True,
        )

    backend._recognize_once = fake_recognize_once

    assert backend.recognize(Image.new("RGB", (2, 2), "white")) == "kept"
    assert prompts == ["OCR:"]
