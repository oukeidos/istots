from __future__ import annotations

from pathlib import Path

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
