from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from istots.app.smoke import (
    SmokeArgumentError,
    SmokeRequest,
    execute_smoke_plan,
    plan_smoke_request,
)
from istots.corrector import CorrectorMode


def test_plan_smoke_request_uses_explicit_output_dir_and_detector_artifact(tmp_path: Path) -> None:
    input_sup = tmp_path / "sample.sup"
    input_sup.write_bytes(b"PG")
    output_dir = tmp_path / "smoke"

    plan = plan_smoke_request(
        SmokeRequest(
            input_sup=input_sup,
            output_dir=output_dir,
        )
    )

    assert plan.output_dir == output_dir.resolve()
    assert plan.is_auto_output_dir is False
    assert plan.convert_plan.input_sup == input_sup.resolve()
    assert plan.convert_plan.output_srt == (output_dir / "sample.smoke.srt").resolve()
    assert plan.convert_plan.detector_output == (output_dir / "sample.detector.jsonl").resolve()
    assert plan.convert_plan.engine == "llama-server"


def test_plan_smoke_request_disables_detector_for_fast_mode(tmp_path: Path) -> None:
    input_sup = tmp_path / "sample.sup"
    input_sup.write_bytes(b"PG")

    plan = plan_smoke_request(
        SmokeRequest(
            input_sup=input_sup,
            output_dir=tmp_path / "smoke",
            ocr_mode="fast",
        )
    )

    assert plan.convert_plan.ocr_mode == "fast"
    assert plan.convert_plan.detector_output is None


def test_plan_smoke_request_builds_corrector_manifest_path(tmp_path: Path) -> None:
    input_sup = tmp_path / "sample.sup"
    input_sup.write_bytes(b"PG")
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"

    plan = plan_smoke_request(
        SmokeRequest(
            input_sup=input_sup,
            output_dir=tmp_path / "smoke",
            corrector="qwen-local",
            corrector_model_path=model_path,
            corrector_mmproj_path=mmproj_path,
        )
    )

    assert plan.convert_plan.corrector_config is not None
    assert plan.convert_plan.corrector_config.mode is CorrectorMode.QWEN_LOCAL
    assert plan.convert_plan.corrector_config.output_path == (
        tmp_path / "smoke" / "sample.corrected.jsonl"
    ).resolve()


def test_plan_smoke_request_uses_injected_tempdir_factory(tmp_path: Path) -> None:
    input_sup = tmp_path / "sample.sup"
    input_sup.write_bytes(b"PG")
    auto_output_dir = tmp_path / "auto-smoke"

    plan = plan_smoke_request(
        SmokeRequest(input_sup=input_sup),
        make_tempdir=lambda prefix: str(auto_output_dir),
    )

    assert plan.output_dir == auto_output_dir.resolve()
    assert plan.is_auto_output_dir is True


def test_plan_smoke_request_rejects_invalid_detector_combo(tmp_path: Path) -> None:
    input_sup = tmp_path / "sample.sup"
    input_sup.write_bytes(b"PG")

    with pytest.raises(SmokeArgumentError) as excinfo:
        plan_smoke_request(
            SmokeRequest(
                input_sup=input_sup,
                no_detector=True,
                detector_mode="wider",
            )
        )

    assert "--detector-mode wider requires detector-enabled smoke validation" in str(excinfo.value)


def test_execute_smoke_plan_removes_auto_output_dir_on_success(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "sample.sup"
    input_sup.write_bytes(b"PG")
    auto_output_dir = tmp_path / "auto-smoke"

    monkeypatch.setattr(
        "istots.app.smoke.execute_convert_plan",
        lambda plan, verbose=True: SimpleNamespace(
            written_count=1,
            output_srt=plan.output_srt,
            device_used="cpu",
            detector_record_count=0,
            correction_record_count=0,
            correction_applied_count=0,
        ),
    )

    plan = plan_smoke_request(
        SmokeRequest(input_sup=input_sup),
        make_tempdir=lambda prefix: str(auto_output_dir),
    )
    result = execute_smoke_plan(plan, verbose=False)

    assert result.removed_output_dir is True
    assert auto_output_dir.exists() is False


def test_execute_smoke_plan_keeps_explicit_output_dir_on_success(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "sample.sup"
    input_sup.write_bytes(b"PG")
    output_dir = tmp_path / "smoke"

    monkeypatch.setattr(
        "istots.app.smoke.execute_convert_plan",
        lambda plan, verbose=True: SimpleNamespace(
            written_count=1,
            output_srt=plan.output_srt,
            device_used="cpu",
            detector_record_count=0,
            correction_record_count=0,
            correction_applied_count=0,
        ),
    )

    plan = plan_smoke_request(
        SmokeRequest(
            input_sup=input_sup,
            output_dir=output_dir,
        )
    )
    result = execute_smoke_plan(plan, verbose=False)

    assert result.removed_output_dir is False
    assert output_dir.exists() is True
