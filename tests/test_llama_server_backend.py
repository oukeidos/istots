from __future__ import annotations

from pathlib import Path

from PIL import Image

from istots.ocr.llama_server_backend import LlamaServerOCRBackend
from istots.llama_runtime import LlamaServerOCRResponse
from istots.ocr.types import LOCAL_PADDLE_MAX_REQUESTS_PER_INSTANCE, resolve_llama_server_request_budget


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
        "istots.ocr.llama_server_backend.request_llama_server_ocr_response",
        lambda spec, image, max_new_tokens, prompt_text="OCR:": LlamaServerOCRResponse(
            text="abc",
            finish_reason="stop",
            completion_tokens=3,
        ),
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


def test_llama_server_backend_applies_length_fallback_for_main_ocr(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")

    process = object()
    prompts: list[str] = []

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

    def fake_request(spec, image, max_new_tokens, prompt_text="OCR:"):
        prompts.append(prompt_text)
        if prompt_text == "OCR:":
            return LlamaServerOCRResponse(
                text="ㄑㄑㄑㄑㄑㄑㄑㄑ",
                finish_reason="length",
                completion_tokens=256,
            )
        return LlamaServerOCRResponse(
            text="good",
            finish_reason="stop",
            completion_tokens=4,
        )

    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.request_llama_server_ocr_response",
        fake_request,
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.stop_llama_server",
        lambda proc: None,
    )

    backend = LlamaServerOCRBackend(max_new_tokens=256, models_dir=tmp_path)
    try:
        assert backend.recognize(Image.new("RGB", (2, 2), "white")) == "good"
    finally:
        backend.close()

    assert prompts == ["OCR:", "Output only the exact text."]


def test_llama_server_backend_skips_length_fallback_outside_main_ocr(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")

    process = object()
    prompts: list[str] = []

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
                "port": 18082,
            },
        )(),
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.start_llama_server",
        lambda spec, startup_timeout_sec: process,
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.request_llama_server_ocr_response",
        lambda spec, image, max_new_tokens, prompt_text="OCR:": (
            prompts.append(prompt_text)
            or LlamaServerOCRResponse(
                text="kept",
                finish_reason="length",
                completion_tokens=256,
            )
        ),
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.stop_llama_server",
        lambda proc: None,
    )

    backend = LlamaServerOCRBackend(max_new_tokens=256, models_dir=tmp_path, role="detector")
    try:
        assert backend.recognize(Image.new("RGB", (2, 2), "white")) == "kept"
    finally:
        backend.close()

    assert prompts == ["OCR:"]


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


def test_llama_server_backend_restarts_after_request_budget(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")

    started: list[object] = []
    stopped: list[object] = []
    processes = [object(), object(), object()]
    request_prompts: list[str] = []

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

    def fake_start(spec, startup_timeout_sec):
        proc = processes[len(started)]
        started.append(proc)
        return proc

    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.start_llama_server",
        fake_start,
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.stop_llama_server",
        lambda proc: stopped.append(proc),
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.request_llama_server_ocr_response",
        lambda spec, image, max_new_tokens, prompt_text="OCR:": (
            request_prompts.append(prompt_text)
            or LlamaServerOCRResponse(
                text="abc",
                finish_reason="stop",
                completion_tokens=3,
            )
        ),
    )

    backend = LlamaServerOCRBackend(
        max_new_tokens=64,
        models_dir=tmp_path,
        max_requests_per_instance=2,
    )
    try:
        images = [Image.new("RGB", (2, 2), "white") for _ in range(5)]
        assert backend.recognize_batch(images) == ["abc", "abc", "abc", "abc", "abc"]
    finally:
        backend.close()

    assert len(started) == 3
    assert stopped == [started[0], started[1], started[2]]
    assert request_prompts == ["OCR:", "OCR:", "OCR:", "OCR:", "OCR:"]


def test_llama_server_backend_ignores_request_budget_for_corrector_role(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "qwen.gguf"
    mmproj = tmp_path / "qwen-mmproj.gguf"
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")

    started: list[object] = []
    stopped: list[object] = []
    process = object()

    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.detect_llama_server_path",
        lambda explicit=None: binary,
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.start_llama_server",
        lambda spec, startup_timeout_sec: (started.append(process) or process),
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.stop_llama_server",
        lambda proc: stopped.append(proc),
    )
    monkeypatch.setattr(
        "istots.ocr.llama_server_backend.request_llama_server_ocr_response",
        lambda spec, image, max_new_tokens, prompt_text="OCR:": LlamaServerOCRResponse(
            text="abc",
            finish_reason="stop",
            completion_tokens=3,
        ),
    )
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MAX_REQUESTS_PER_INSTANCE", "2")

    backend = LlamaServerOCRBackend(
        role="corrector",
        model_path=model,
        mmproj_path=mmproj,
        max_requests_per_instance=2,
    )
    try:
        images = [Image.new("RGB", (2, 2), "white") for _ in range(5)]
        assert backend.recognize_batch(images) == ["abc", "abc", "abc", "abc", "abc"]
        assert backend.max_requests_per_instance is None
    finally:
        backend.close()

    assert started == [process]
    assert stopped == [process]


def test_llama_server_backend_applies_default_request_budget_for_paddle_roles(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")

    process = object()

    monkeypatch.delenv("ISTOTS_LLAMA_SERVER_MAX_REQUESTS_PER_INSTANCE", raising=False)
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
        "istots.ocr.llama_server_backend.stop_llama_server",
        lambda proc: None,
    )

    backend = LlamaServerOCRBackend(models_dir=tmp_path)
    try:
        assert backend.max_requests_per_instance == LOCAL_PADDLE_MAX_REQUESTS_PER_INSTANCE
    finally:
        backend.close()


def test_resolve_llama_server_request_budget_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MAX_REQUESTS_PER_INSTANCE", "100")
    assert resolve_llama_server_request_budget(None) == 100
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MAX_REQUESTS_PER_INSTANCE", "0")
    assert resolve_llama_server_request_budget(None) is None
    monkeypatch.delenv("ISTOTS_LLAMA_SERVER_MAX_REQUESTS_PER_INSTANCE", raising=False)
    assert (
        resolve_llama_server_request_budget(
            None,
            default=LOCAL_PADDLE_MAX_REQUESTS_PER_INSTANCE,
        )
        == 200
    )
