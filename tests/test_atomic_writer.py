from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from istots import atomic_writer, corrector, detector, gemini_auth, srt_writer
from istots.corrector import ConservativeCorrectionRecord
from istots.detector import HybridDetectorRecord
from istots.srt_writer import SubtitleEntry


def _patch_atomic_write_text_file_failure(monkeypatch) -> None:
    def failing_atomic_write_text_file(
        path: Path,
        write_text,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.forced-failure.tmp")
        try:
            with temp_path.open("w", encoding=encoding, newline=newline) as handle:
                write_text(handle)
                handle.flush()
            raise RuntimeError("forced atomic write failure")
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    monkeypatch.setattr(atomic_writer, "atomic_write_text_file", failing_atomic_write_text_file)


def _detector_record(index: int) -> HybridDetectorRecord:
    return HybridDetectorRecord(
        index=index,
        raw_index=index,
        window_id=0,
        start_ms=index * 1000,
        end_ms=index * 1000 + 500,
        detector_branch="branch",
        shape="wide",
        ratio=1.0,
        option_role="alternate",
        baseline_text=f"baseline-{index}",
        option_text=f"option-{index}",
        diff_label="changed",
        meaningful=True,
        char_error_rate=0.1,
    )


def _correction_record(index: int) -> ConservativeCorrectionRecord:
    return ConservativeCorrectionRecord(
        index=index,
        raw_index=index,
        window_id=0,
        start_ms=index * 1000,
        end_ms=index * 1000 + 500,
        detector_branch="branch",
        shape="wide",
        ratio=1.0,
        option_role="alternate",
        baseline_text=f"baseline-{index}",
        option_text=f"option-{index}",
        diff_label="changed",
        meaningful=True,
        char_error_rate=0.1,
        anchor_count=1,
        corrector_name="gemini-test",
        corrector_prompt_style="strict_ocr_v1",
        corrector_text=f"corrected-{index}",
        conservative_merged_text=f"merged-{index}",
        applied_op_count=1,
        raw_changed=True,
        merged_changed=True,
        corrector_reasoning_content="reasoning",
    )


def test_atomic_write_text_file_preserves_existing_file_and_cleans_temp_on_failure(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "artifact.txt"
    output_path.write_text("OLD\n", encoding="utf-8")

    def write_partial(handle) -> None:
        handle.write("NEW\n")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        atomic_writer.atomic_write_text_file(output_path, write_partial)

    assert output_path.read_text(encoding="utf-8") == "OLD\n"
    assert list(tmp_path.glob(".artifact.txt.*.tmp")) == []


def test_atomic_write_json_preserves_formatting(tmp_path: Path) -> None:
    output_path = tmp_path / "payload.json"

    atomic_writer.atomic_write_json(output_path, {"message": "한글"}, ensure_ascii=False, indent=2)

    assert output_path.read_text(encoding="utf-8") == '{\n  "message": "한글"\n}\n'


def test_atomic_write_jsonl_preserves_formatting(tmp_path: Path) -> None:
    output_path = tmp_path / "payload.jsonl"

    atomic_writer.atomic_write_jsonl(
        output_path,
        [{"message": "한글"}, {"count": 2}],
        ensure_ascii=False,
    )

    assert output_path.read_text(encoding="utf-8") == '{"message": "한글"}\n{"count": 2}\n'


def test_write_srt_preserves_existing_file_on_mid_write_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "sample.srt"
    output_path.write_text("OLD-SRT\n", encoding="utf-8")
    entries = [
        SubtitleEntry(index=1, start=timedelta(seconds=0), end=timedelta(seconds=1), text="first"),
        SubtitleEntry(index=2, start=timedelta(seconds=2), end=timedelta(seconds=3), text="second"),
    ]
    original_format_timestamp = srt_writer.format_timestamp
    calls = {"count": 0}

    def failing_format_timestamp(ts: timedelta) -> str:
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("boom")
        return original_format_timestamp(ts)

    monkeypatch.setattr(srt_writer, "format_timestamp", failing_format_timestamp)

    with pytest.raises(RuntimeError, match="boom"):
        srt_writer.write_srt(entries, output_path)

    assert output_path.read_text(encoding="utf-8") == "OLD-SRT\n"


def test_write_hybrid_detector_records_preserves_existing_file_on_mid_write_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "detector.jsonl"
    output_path.write_text("OLD-DET\n", encoding="utf-8")
    original_asdict = detector.asdict
    calls = {"count": 0}

    def failing_asdict(record) -> dict:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")
        return original_asdict(record)

    monkeypatch.setattr(detector, "asdict", failing_asdict)

    with pytest.raises(RuntimeError, match="boom"):
        detector.write_hybrid_detector_records(output_path, [_detector_record(1), _detector_record(2)])

    assert output_path.read_text(encoding="utf-8") == "OLD-DET\n"


def test_write_correction_records_preserves_existing_file_on_mid_write_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "corrected.jsonl"
    output_path.write_text("OLD-CORR\n", encoding="utf-8")
    original_asdict = corrector.asdict
    calls = {"count": 0}

    def failing_asdict(record) -> dict:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")
        return original_asdict(record)

    monkeypatch.setattr(corrector, "asdict", failing_asdict)

    with pytest.raises(RuntimeError, match="boom"):
        corrector.write_correction_records(output_path, [_correction_record(1), _correction_record(2)])

    assert output_path.read_text(encoding="utf-8") == "OLD-CORR\n"


def test_save_cached_item_preserves_existing_file_when_atomic_write_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "cache-key.json"
    cache_path.write_text('{\n  "cached": "OLD"\n}\n', encoding="utf-8")
    _patch_atomic_write_text_file_failure(monkeypatch)

    with pytest.raises(RuntimeError, match="forced atomic write failure"):
        corrector._save_cached_item(  # noqa: SLF001
            cache_dir,
            "cache-key",
            {"cached": "NEW"},
        )

    assert cache_path.read_text(encoding="utf-8") == '{\n  "cached": "OLD"\n}\n'


def test_write_auth_config_preserves_existing_file_when_atomic_write_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "auth.json"
    config_path.write_text('{\n  "gemini_env_file_path": "/old.env"\n}\n', encoding="utf-8")
    monkeypatch.setenv("ISTOTS_AUTH_CONFIG_PATH", str(config_path))
    _patch_atomic_write_text_file_failure(monkeypatch)

    with pytest.raises(RuntimeError, match="forced atomic write failure"):
        gemini_auth._write_auth_config({"gemini_env_file_path": "/new.env"})  # noqa: SLF001

    assert json.loads(config_path.read_text(encoding="utf-8")) == {"gemini_env_file_path": "/old.env"}
