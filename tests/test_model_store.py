from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from istots import model_store


def test_model_dir_name_replaces_slash() -> None:
    assert model_store.model_dir_name("PaddlePaddle/PaddleOCR-VL-1.5") == "PaddlePaddle__PaddleOCR-VL-1.5"


def test_resolve_local_model_path_uses_cache_root(tmp_path: Path) -> None:
    path = model_store.resolve_local_model_path("org/model", models_dir=tmp_path)
    assert path == (tmp_path / "org__model").resolve()


def test_ensure_local_model_accepts_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "local_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    resolved = model_store.ensure_local_model(str(model_dir))
    assert resolved == model_dir.resolve()


def test_ensure_local_model_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Model is not available locally"):
        model_store.ensure_local_model("org/model", models_dir=tmp_path)


def test_download_gguf_runtime_assets_requests_expected_files(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    target = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    target.mkdir(parents=True, exist_ok=True)
    (target / model_store.DEFAULT_GGUF_FILENAME).write_bytes(b"model")
    (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).write_bytes(b"mmproj")

    def fake_snapshot_download(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    gguf_dir, model_path, mmproj_path = model_store.download_gguf_runtime_assets(models_dir=tmp_path)

    assert gguf_dir == target.resolve()
    assert model_path == (target / model_store.DEFAULT_GGUF_FILENAME).resolve()
    assert mmproj_path == (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).resolve()
    assert captured["repo_id"] == model_store.DEFAULT_GGUF_MODEL_ID
    assert captured["allow_patterns"] == [
        model_store.DEFAULT_GGUF_FILENAME,
        model_store.DEFAULT_GGUF_MMPROJ_FILENAME,
        "LICENSE*",
        "README*",
        "*.json",
    ]


def test_setup_default_runtime_assets_downloads_and_materializes(monkeypatch, tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf_model"
    gguf_dir = tmp_path / "gguf_model"
    gguf_model_path = gguf_dir / model_store.DEFAULT_GGUF_FILENAME
    gguf_mmproj_path = gguf_dir / model_store.DEFAULT_GGUF_MMPROJ_FILENAME
    derived_path = gguf_dir / "PaddleOCR-VL-1.5-mmproj.minpix32768.gguf"

    monkeypatch.setattr(
        model_store,
        "download_model",
        lambda model_id, models_dir=None, force=False: hf_dir,
    )
    monkeypatch.setattr(
        model_store,
        "download_gguf_runtime_assets",
        lambda model_id=model_store.DEFAULT_GGUF_MODEL_ID, models_dir=None, force=False: (
            gguf_dir,
            gguf_model_path,
            gguf_mmproj_path,
        ),
    )

    calls: dict[str, object] = {}

    def fake_materialize_mmproj(
        *,
        base_mmproj,
        min_pixels,
        support_dir=None,
        gguf_py_base_url=None,
        gguf_source_mode="auto",
        force=False,
        output_path=None,
    ):
        calls["base_mmproj"] = base_mmproj
        calls["min_pixels"] = min_pixels
        calls["gguf_source_mode"] = gguf_source_mode
        calls["force"] = force
        return derived_path

    monkeypatch.setattr(
        "istots.llama_mmproj.materialize_mmproj",
        fake_materialize_mmproj,
    )

    artifacts = model_store.setup_default_runtime_assets(
        models_dir=tmp_path,
        force=True,
        gguf_source_mode="auto",
    )

    assert artifacts.hf_model_dir == hf_dir
    assert artifacts.gguf_model_dir == gguf_dir
    assert artifacts.gguf_model_path == gguf_model_path
    assert artifacts.gguf_mmproj_path == gguf_mmproj_path
    assert artifacts.gguf_mmproj_minpix32768_path == derived_path
    assert calls == {
        "base_mmproj": gguf_mmproj_path,
        "min_pixels": 32768,
        "gguf_source_mode": "auto",
        "force": True,
    }
