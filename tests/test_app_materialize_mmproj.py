from __future__ import annotations

from pathlib import Path

from istots.app.materialize_mmproj import (
    MaterializeMmprojRequest,
    execute_materialize_mmproj_request,
)


def test_execute_materialize_mmproj_request_returns_output_and_applied_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    output = tmp_path / "derived.gguf"

    monkeypatch.setattr(
        "istots.app.materialize_mmproj.llama_mmproj_module.materialize_mmproj",
        lambda **kwargs: captured.update({"materialize": kwargs}) or output,
    )
    monkeypatch.setattr(
        "istots.app.materialize_mmproj.llama_mmproj_module.read_mmproj_min_pixels",
        lambda *args, **kwargs: captured.update({"read": {"args": args, "kwargs": kwargs}}) or 32768,
    )

    result = execute_materialize_mmproj_request(
        MaterializeMmprojRequest(
            base_mmproj=Path("base.gguf"),
            output=output,
            gguf_source_mode="installed",
        )
    )

    assert result.output == output
    assert result.applied_value == 32768
    assert captured["materialize"]["base_mmproj"] == Path("base.gguf")
    assert captured["materialize"]["output_path"] == output
    assert captured["materialize"]["gguf_source_mode"] == "installed"
    assert captured["read"]["args"] == (output,)
