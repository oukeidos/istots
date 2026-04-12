from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from istots import llama_mmproj


def test_default_materialized_mmproj_path() -> None:
    path = Path("/tmp/PaddleOCR-VL-1.5-mmproj.gguf")
    assert llama_mmproj.default_materialized_mmproj_path(path, 32768) == Path(
        "/tmp/PaddleOCR-VL-1.5-mmproj.minpix32768.gguf"
    )


def test_materialize_mmproj_rejects_in_place(tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"gguf")
    with pytest.raises(RuntimeError, match="output path must differ"):
        llama_mmproj.materialize_mmproj(
            base_mmproj=base,
            output_path=base,
        )


def test_materialize_mmproj_uses_known_good_gguf(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"base-bytes")
    output = tmp_path / "derived.gguf"

    calls: dict[str, object] = {}
    field = SimpleNamespace(
        parts=[np.array([112896], dtype=np.uint32)],
        data=[0],
        types=["fake_uint32"],
    )

    class FakeMemMap:
        def flush(self) -> None:
            calls["flushed"] = True

    class FakeReader:
        gguf_scalar_to_np = {"fake_uint32": np.uint32}

        def __init__(self, path: str, mode: str) -> None:
            calls.setdefault("paths", []).append((path, mode))
            self.data = FakeMemMap()

        def get_field(self, key: str):
            calls.setdefault("keys", []).append(key)
            return field

    fake_gguf = SimpleNamespace(GGUFReader=FakeReader)

    monkeypatch.setattr(
        llama_mmproj,
        "load_known_good_gguf",
        lambda source_mode="auto", support_dir=None, base_url=None: fake_gguf,
    )

    result = llama_mmproj.materialize_mmproj(
        base_mmproj=base,
        output_path=output,
        min_pixels=32768,
        gguf_py_base_url=str(tmp_path / "source"),
    )

    assert result == output.resolve()
    assert output.read_bytes() == b"base-bytes"
    assert int(field.parts[0][0]) == 32768
    assert calls["keys"] == [llama_mmproj.MMPROJ_MIN_PIXELS_KEY, llama_mmproj.MMPROJ_MIN_PIXELS_KEY]
    assert calls["flushed"] is True
