from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from istots.app.setup import SetupArgumentError, SetupRequest, execute_setup_request


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
    assert result.artifacts is artifacts
    assert result.custom_hf_bundle is True
    assert result.custom_gguf_bundle is True
    assert result.custom_qwen_bundle is True
