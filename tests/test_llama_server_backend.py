from __future__ import annotations

from pathlib import Path

from PIL import Image

from istots.ocr.llama_server_backend import LlamaServerOCRBackend


def test_llama_server_backend_starts_runtime_and_recognizes(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")

    process = object()
    stopped: list[object] = []

    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.detect_llama_server_path",
        lambda explicit=None: binary,
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.build_llama_server_launch_spec",
        lambda **kwargs: type(
            "Spec",
            (),
            {
                "model_path": model,
                "mmproj_path": mmproj,
                "host": "127.0.0.1",
                "port": 18080,
            },
        )(),
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.start_llama_server",
        lambda spec, startup_timeout_sec: process,
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.request_llama_server_ocr",
        lambda spec, image, max_new_tokens, prompt_text="OCR:": "abc",
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.stop_llama_server",
        lambda proc: stopped.append(proc),
    )

    backend = LlamaServerOCRBackend(max_new_tokens=64, models_dir=tmp_path)
    try:
        assert backend.recognize_batch([Image.new("RGB", (2, 2), "white")]) == ["abc"]
    finally:
        backend.close()

    assert stopped == [process]


def test_llama_server_backend_rejects_missing_assets(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")

    missing_model = tmp_path / "missing-model.gguf"
    missing_mmproj = tmp_path / "missing-mmproj.gguf"

    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.detect_llama_server_path",
        lambda explicit=None: binary,
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.build_llama_server_launch_spec",
        lambda **kwargs: type(
            "Spec",
            (),
            {
                "model_path": missing_model,
                "mmproj_path": missing_mmproj,
                "host": "127.0.0.1",
                "port": 18080,
            },
        )(),
    )

    try:
        LlamaServerOCRBackend(models_dir=tmp_path)
    except RuntimeError as exc:
        assert "required llama-server runtime assets are missing" in str(exc)
    else:
        raise AssertionError("expected missing asset error")
