from __future__ import annotations

import builtins

import pytest

from istots.ocr.hf_backend import HFPaddleOCRVLBackend


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
