from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from istots.app.convert import (
    ConvertArgumentError,
    ConvertProgressEstimator,
    ConvertProgressEvent,
    ConvertRequest,
    ConvertResult,
    execute_convert_plan,
    plan_convert_request,
)
from istots.corrector import CorrectorMode
from istots.ocr import PaddleOCRVLRuntimeOverrides


def test_plan_convert_request_resolves_hf_model_and_normalizes_paths(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    model_dir = tmp_path / "cached-model"
    model_dir.mkdir()

    monkeypatch.setattr(
        "istots.app.convert.model_store.ensure_local_model",
        lambda model_id, models_dir=None: model_dir,
    )

    plan = plan_convert_request(
        ConvertRequest(
            input_sup=input_sup,
            output_srt=output_srt,
            engine="hf",
            model_id="org/model",
        )
    )

    assert plan.input_sup == input_sup.resolve()
    assert plan.output_srt == output_srt.resolve()
    assert plan.model_id == str(model_dir)
    assert plan.resolved_hf_model_path == model_dir
    assert plan.local_files_only is True


def test_plan_convert_request_reports_existing_output_artifacts(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    output_srt.write_text("existing", encoding="utf-8")
    detector_output.write_text("existing", encoding="utf-8")

    plan = plan_convert_request(
        ConvertRequest(
            input_sup=input_sup,
            output_srt=output_srt,
            detector_output=detector_output,
        )
    )

    assert plan.existing_output_artifacts == (
        output_srt.resolve(),
        detector_output.resolve(),
    )


def test_plan_convert_request_builds_qwen_local_corrector_config(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    corrector_output = tmp_path / "corrected.jsonl"
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "qwen-mmproj.gguf"

    plan = plan_convert_request(
        ConvertRequest(
            input_sup=input_sup,
            output_srt=output_srt,
            corrector="qwen-local",
            corrector_model_path=model_path,
            corrector_mmproj_path=mmproj_path,
            corrector_output=corrector_output,
            qwen_profile="cpu",
            qwen_port=19007,
        )
    )

    assert plan.corrector_config is not None
    assert plan.corrector_config.mode is CorrectorMode.QWEN_LOCAL
    assert plan.corrector_config.output_path == corrector_output.resolve()
    assert plan.corrector_config.local_model_path == model_path.resolve()
    assert plan.corrector_config.local_mmproj_path == mmproj_path.resolve()
    assert plan.corrector_config.local_runtime_overrides.port == 19007
    assert plan.corrector_config.local_runtime_overrides.profile == "cpu"


def test_plan_convert_request_resolves_default_qwen_corrector_assets(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    model_path = tmp_path / "cached-qwen.gguf"
    mmproj_path = tmp_path / "cached-qwen-mmproj.gguf"

    monkeypatch.setattr(
        "istots.app.convert.model_store.ensure_local_qwen_corrector_assets",
        lambda models_dir=None: (model_path, mmproj_path),
    )

    plan = plan_convert_request(
        ConvertRequest(
            input_sup=input_sup,
            output_srt=output_srt,
            corrector="qwen-local",
        )
    )

    assert plan.corrector_config is not None
    assert plan.corrector_config.local_model_path == model_path
    assert plan.corrector_config.local_mmproj_path == mmproj_path


def test_plan_convert_request_rejects_duplicate_paths(tmp_path: Path) -> None:
    input_sup = tmp_path / "same.sup"
    input_sup.write_bytes(b"PG")

    with pytest.raises(ConvertArgumentError) as excinfo:
        plan_convert_request(
            ConvertRequest(
                input_sup=input_sup,
                output_srt=input_sup,
            )
        )

    assert "input_sup and output_srt must be different paths" in str(excinfo.value)


def test_execute_convert_plan_calls_pipeline(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "istots.app.convert.pipeline.convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs)
        or SimpleNamespace(
            written_count=1,
            output_srt=output_srt,
            device_used="cpu",
            detector_record_count=0,
            correction_record_count=0,
            correction_applied_count=0,
        ),
    )

    plan = plan_convert_request(
        ConvertRequest(
            input_sup=input_sup,
            output_srt=output_srt,
            paddle_profile="cpu",
            paddle_port=19005,
        )
    )
    result = execute_convert_plan(plan, verbose=False)

    assert isinstance(result, ConvertResult)
    assert result.output_srt == output_srt
    assert captured["input_sup"] == input_sup.resolve()
    assert captured["output_srt"] == output_srt.resolve()
    assert captured["engine"] == "llama-server"
    assert captured["verbose"] is False
    assert captured["progress_callback"] is None
    assert captured["paddle_runtime_overrides"] == PaddleOCRVLRuntimeOverrides(
        profile="cpu",
        port=19005,
        threads=None,
        threads_batch=None,
        gpu_layers=None,
        no_mmproj_offload=None,
        startup_timeout_sec=120.0,
        ctx_size=None,
    )


def test_convert_progress_estimator_aggregates_fast_and_tall_rows(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"x" * 220_000)

    estimator = ConvertProgressEstimator(
        input_sup=input_sup,
        enable_furigana_mask=False,
        ocr_mode="fast",
    )

    estimator.record(ConvertProgressEvent(phase="prepare_started", elapsed_sec=0.0))
    estimator.record(ConvertProgressEvent(phase="prepare_completed", elapsed_sec=2.0, total_rows=60))
    estimator.record(
        ConvertProgressEvent(
            phase="partition_completed",
            elapsed_sec=2.0,
            total_rows=60,
            wide_total_rows=48,
            tall_total_rows=12,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="backend_loading",
            elapsed_sec=2.1,
            branch_label="non-tall-fast",
            role="ocr-fast",
            branch_total_rows=48,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="backend_ready",
            elapsed_sec=6.1,
            branch_label="non-tall-fast",
            role="ocr-fast",
            branch_total_rows=48,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="ocr_started",
            elapsed_sec=6.1,
            branch_label="non-tall-fast",
            branch_total_rows=48,
            branch_total_unique=40,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="ocr_progress",
            elapsed_sec=16.1,
            branch_label="non-tall-fast",
            branch_total_rows=48,
            branch_processed_rows=24,
            branch_total_unique=40,
            branch_processed_unique=20,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="backend_loading",
            elapsed_sec=16.2,
            branch_label="tall-default",
            role="ocr",
            branch_total_rows=12,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="backend_ready",
            elapsed_sec=20.2,
            branch_label="tall-default",
            role="ocr",
            branch_total_rows=12,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="ocr_started",
            elapsed_sec=20.2,
            branch_label="tall-default",
            branch_total_rows=12,
            branch_total_unique=10,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="ocr_progress",
            elapsed_sec=24.2,
            branch_label="tall-default",
            branch_total_rows=12,
            branch_processed_rows=6,
            branch_total_unique=10,
            branch_processed_unique=5,
        )
    )

    snapshot = estimator.snapshot(now_monotonic=estimator._clock_base + 24.2)

    assert snapshot.headline == "OCR"
    assert snapshot.detail == "30/60"
    assert 0.5 < snapshot.fraction < 0.95
    assert snapshot.eta_sec is not None
    assert snapshot.eta_sec > 0.0


def test_convert_progress_estimator_uses_partition_unique_counts_for_future_tall_eta(
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"x" * 220_000)

    with_unique = ConvertProgressEstimator(
        input_sup=input_sup,
        enable_furigana_mask=False,
        ocr_mode="fast",
    )
    without_unique = ConvertProgressEstimator(
        input_sup=input_sup,
        enable_furigana_mask=False,
        ocr_mode="fast",
    )

    common_events = [
        ConvertProgressEvent(phase="prepare_started", elapsed_sec=0.0),
        ConvertProgressEvent(phase="prepare_completed", elapsed_sec=2.0, total_rows=60),
        ConvertProgressEvent(
            phase="backend_loading",
            elapsed_sec=2.1,
            branch_label="non-tall-fast",
            role="ocr-fast",
            branch_total_rows=48,
            branch_total_unique=40,
        ),
        ConvertProgressEvent(
            phase="backend_ready",
            elapsed_sec=6.1,
            branch_label="non-tall-fast",
            role="ocr-fast",
            branch_total_rows=48,
            branch_total_unique=40,
        ),
        ConvertProgressEvent(
            phase="ocr_started",
            elapsed_sec=6.1,
            branch_label="non-tall-fast",
            branch_total_rows=48,
            branch_total_unique=40,
        ),
        ConvertProgressEvent(
            phase="ocr_progress",
            elapsed_sec=16.1,
            branch_label="non-tall-fast",
            branch_total_rows=48,
            branch_processed_rows=24,
            branch_total_unique=40,
            branch_processed_unique=20,
        ),
    ]

    with_unique.record(
        ConvertProgressEvent(
            phase="partition_completed",
            elapsed_sec=2.0,
            total_rows=60,
            wide_total_rows=48,
            tall_total_rows=12,
            wide_total_unique=40,
            tall_total_unique=4,
        )
    )
    without_unique.record(
        ConvertProgressEvent(
            phase="partition_completed",
            elapsed_sec=2.0,
            total_rows=60,
            wide_total_rows=48,
            tall_total_rows=12,
        )
    )

    for event in common_events:
        with_unique.record(event)
        without_unique.record(event)

    with_snapshot = with_unique.snapshot(now_monotonic=with_unique._clock_base + 16.1)
    without_snapshot = without_unique.snapshot(now_monotonic=without_unique._clock_base + 16.1)

    assert with_snapshot.eta_sec is not None
    assert without_snapshot.eta_sec is not None
    assert with_snapshot.eta_sec < without_snapshot.eta_sec


def test_convert_progress_estimator_predicts_future_tall_rate_from_wide_speed(
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"x" * 220_000)

    estimator = ConvertProgressEstimator(
        input_sup=input_sup,
        enable_furigana_mask=False,
        ocr_mode="fast",
    )

    estimator.record(ConvertProgressEvent(phase="prepare_started", elapsed_sec=0.0))
    estimator.record(ConvertProgressEvent(phase="prepare_completed", elapsed_sec=2.0, total_rows=60))
    estimator.record(
        ConvertProgressEvent(
            phase="partition_completed",
            elapsed_sec=2.0,
            total_rows=60,
            wide_total_rows=48,
            tall_total_rows=12,
            wide_total_unique=40,
            tall_total_unique=10,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="ocr_started",
            elapsed_sec=6.0,
            branch_label="non-tall-fast",
            branch_total_rows=48,
            branch_total_unique=40,
        )
    )
    estimator.record(
        ConvertProgressEvent(
            phase="ocr_progress",
            elapsed_sec=10.0,
            branch_label="non-tall-fast",
            branch_total_rows=48,
            branch_processed_rows=10,
            branch_total_unique=40,
            branch_processed_unique=8,
        )
    )

    assert estimator._branch_rate_prior_sec("tall-default") == pytest.approx(0.8)
