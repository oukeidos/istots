from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from istots.app.setup import SetupArgumentError, SetupProgressEvent, SetupRequest, execute_setup_request


def test_execute_setup_request_rejects_custom_hf_model_without_opt_in() -> None:
    with pytest.raises(SetupArgumentError) as excinfo:
        execute_setup_request(SetupRequest(model_id="custom/hf"))

    assert "--model-id requires --with-hf-fallback" in str(excinfo.value)


def test_execute_setup_request_returns_artifacts_and_custom_bundle_flags(monkeypatch, tmp_path: Path) -> None:
    artifacts = SimpleNamespace(
        hf_model_dir=tmp_path / "hf_model",
        gguf_model_dir=tmp_path / "gguf_model",
        gguf_model_path=tmp_path / "gguf_model" / "custom.gguf",
        gguf_mmproj_path=tmp_path / "gguf_model" / "custom-mmproj.gguf",
        gguf_mmproj_minpix32768_path=tmp_path / "gguf_model" / "custom-mmproj.minpix32768.gguf",
        qwen_corrector_dir=tmp_path / "qwen_model",
        qwen_corrector_model_path=tmp_path / "qwen_model" / "custom-qwen.gguf",
        qwen_corrector_mmproj_path=tmp_path / "qwen_model" / "custom-qwen-mmproj.gguf",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "istots.app.setup.model_store.setup_default_runtime_assets",
        lambda **kwargs: captured.update(kwargs) or artifacts,
    )

    result = execute_setup_request(
        SetupRequest(
            with_hf_fallback=True,
            model_id="custom/hf",
            gguf_model_id="custom/gguf",
            with_qwen_corrector=True,
            qwen_corrector_model_id="custom/qwen",
            qwen_corrector_model_filename="custom-qwen.gguf",
            qwen_corrector_mmproj_filename="custom-qwen-mmproj.gguf",
        )
    )

    assert captured["hf_model_id"] == "custom/hf"
    assert captured["gguf_model_id"] == "custom/gguf"
    assert captured["cancel_callback"] is None
    assert result.artifacts is artifacts
    assert result.custom_hf_bundle is True
    assert result.custom_gguf_bundle is True
    assert result.custom_qwen_bundle is True


def test_execute_setup_request_bootstraps_managed_runtime_and_emits_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifacts = SimpleNamespace(
        hf_model_dir=None,
        gguf_model_dir=tmp_path / "gguf_model",
        gguf_model_path=tmp_path / "gguf_model" / "custom.gguf",
        gguf_mmproj_path=tmp_path / "gguf_model" / "custom-mmproj.gguf",
        gguf_mmproj_minpix32768_path=tmp_path / "derived" / "custom-mmproj.minpix32768.gguf",
        qwen_corrector_dir=None,
        qwen_corrector_model_path=None,
        qwen_corrector_mmproj_path=None,
    )
    bootstrapped: list[dict[str, object]] = []
    setup_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "istots.gui.bootstrap_windows.install_managed_llama_cpp_runtime",
        lambda **kwargs: bootstrapped.append(kwargs),
    )
    monkeypatch.setattr(
        "istots.app.setup.model_store.setup_default_runtime_assets",
        lambda **kwargs: setup_calls.append(kwargs) or artifacts,
    )

    progress_events: list[SetupProgressEvent] = []
    result = execute_setup_request(
        SetupRequest(
            models_dir=tmp_path / "models",
            derived_mmproj_output_path=tmp_path / "derived" / "custom-mmproj.minpix32768.gguf",
            bootstrap_managed_runtime=True,
            runtime_variant="auto",
            install_prerequisites=True,
        ),
        progress_callback=progress_events.append,
    )

    assert result.artifacts is artifacts
    assert bootstrapped and bootstrapped[0]["requested_variant"] == "auto"
    assert bootstrapped[0]["install_prerequisites"] is True
    assert bootstrapped[0]["cancel_event"] is None
    assert setup_calls and setup_calls[0]["derived_mmproj_output_path"] == (
        tmp_path / "derived" / "custom-mmproj.minpix32768.gguf"
    )
    assert [event.phase for event in progress_events][-2:] == ["model_setup", "complete"]


def test_execute_setup_request_records_diagnostic_events(monkeypatch, tmp_path: Path) -> None:
    artifacts = SimpleNamespace(
        hf_model_dir=None,
        gguf_model_dir=tmp_path / "gguf_model",
        gguf_model_path=tmp_path / "gguf_model" / "custom.gguf",
        gguf_mmproj_path=tmp_path / "gguf_model" / "custom-mmproj.gguf",
        gguf_mmproj_minpix32768_path=tmp_path / "derived" / "custom-mmproj.minpix32768.gguf",
        qwen_corrector_dir=None,
        qwen_corrector_model_path=None,
        qwen_corrector_mmproj_path=None,
    )
    seen: list[str] = []

    monkeypatch.setattr(
        "istots.app.setup.append_runtime_diagnostic_event",
        lambda event, **kwargs: seen.append(event),
    )
    monkeypatch.setattr(
        "istots.app.setup.model_store.setup_default_runtime_assets",
        lambda **kwargs: artifacts,
    )

    result = execute_setup_request(SetupRequest(models_dir=tmp_path / "models"))

    assert result.artifacts is artifacts
    assert seen == [
        "setup_request_start",
        "setup_model_assets_start",
        "setup_model_assets_complete",
        "setup_request_complete",
    ]


def test_execute_setup_request_records_diagnostic_error(monkeypatch, tmp_path: Path) -> None:
    seen: list[str] = []

    monkeypatch.setattr(
        "istots.app.setup.append_runtime_diagnostic_event",
        lambda event, **kwargs: seen.append(event),
    )
    monkeypatch.setattr(
        "istots.app.setup.model_store.setup_default_runtime_assets",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(Exception, match="boom"):
        execute_setup_request(SetupRequest(models_dir=tmp_path / "models"))

    assert seen == [
        "setup_request_start",
        "setup_model_assets_start",
        "setup_request_error",
    ]


def test_execute_setup_request_can_skip_completion_event(monkeypatch, tmp_path: Path) -> None:
    artifacts = SimpleNamespace(
        hf_model_dir=None,
        gguf_model_dir=tmp_path / "gguf_model",
        gguf_model_path=tmp_path / "gguf_model" / "custom.gguf",
        gguf_mmproj_path=tmp_path / "gguf_model" / "custom-mmproj.gguf",
        gguf_mmproj_minpix32768_path=tmp_path / "derived" / "custom-mmproj.minpix32768.gguf",
        qwen_corrector_dir=None,
        qwen_corrector_model_path=None,
        qwen_corrector_mmproj_path=None,
    )

    monkeypatch.setattr(
        "istots.app.setup.model_store.setup_default_runtime_assets",
        lambda **kwargs: artifacts,
    )

    progress_events: list[SetupProgressEvent] = []
    execute_setup_request(
        SetupRequest(models_dir=tmp_path / "models"),
        progress_callback=progress_events.append,
        emit_completion_event=False,
    )

    assert [event.phase for event in progress_events] == ["model_setup"]


def test_execute_setup_request_cancels_before_model_setup(monkeypatch, tmp_path: Path) -> None:
    cancel_event = threading.Event()
    cancel_event.set()

    monkeypatch.setattr(
        "istots.app.setup.model_store.setup_default_runtime_assets",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("setup assets should not start after cancellation")),
    )

    with pytest.raises(Exception, match="setup cancelled during startup"):
        execute_setup_request(
            SetupRequest(models_dir=tmp_path / "models"),
            cancel_event=cancel_event,
        )
