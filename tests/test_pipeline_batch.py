from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from istots import pipeline


def test_is_oom_error_detects_common_message() -> None:
    exc = RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
    assert pipeline._is_oom_error(exc)  # noqa: SLF001


def test_is_oom_error_detects_exception_name() -> None:
    class FakeOutOfMemoryError(Exception):
        pass

    assert pipeline._is_oom_error(FakeOutOfMemoryError("boom"))  # noqa: SLF001


def test_is_oom_error_ignores_unrelated_error() -> None:
    exc = RuntimeError("model input shape mismatch")
    assert not pipeline._is_oom_error(exc)  # noqa: SLF001


def test_convert_sup_to_srt_releases_backend_on_success(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    class FakeBackend:
        instances: list["FakeBackend"] = []

        def __init__(self, **kwargs) -> None:
            self.closed = False
            FakeBackend.instances.append(self)

        def recognize_batch(self, images):
            return []

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    def fake_iter_sup_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](0)
        return iter([])

    def fake_write_srt(entries, path):
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(pipeline, "resolve_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "HFPaddleOCRVLBackend", FakeBackend)
    monkeypatch.setattr(pipeline, "iter_sup_frames", fake_iter_sup_frames)
    monkeypatch.setattr(pipeline, "write_srt", fake_write_srt)

    result = pipeline.convert_sup_to_srt(
        input_sup=input_sup,
        output_srt=output_srt,
        verbose=False,
    )

    assert result.written_count == 0
    assert FakeBackend.instances[0].closed


def test_convert_sup_to_srt_releases_backend_on_error(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    output_srt = tmp_path / "output.srt"
    input_sup.write_bytes(b"")

    frame = SimpleNamespace(
        start=timedelta(milliseconds=0),
        end=timedelta(milliseconds=10),
        image=Image.new("RGB", (2, 2), "white"),
    )

    class FakeBackend:
        instances: list["FakeBackend"] = []

        def __init__(self, **kwargs) -> None:
            self.closed = False
            FakeBackend.instances.append(self)

        def recognize_batch(self, images):
            raise RuntimeError("boom")

        def clear_device_cache(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    def fake_iter_sup_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](1)
        return iter([frame])

    monkeypatch.setattr(pipeline, "resolve_device", lambda preferred_device: "cpu")
    monkeypatch.setattr(pipeline, "HFPaddleOCRVLBackend", FakeBackend)
    monkeypatch.setattr(pipeline, "iter_sup_frames", fake_iter_sup_frames)
    monkeypatch.setattr(pipeline, "write_srt", lambda entries, path: None)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            batch_size=1,
            verbose=False,
        )

    assert FakeBackend.instances[0].closed
