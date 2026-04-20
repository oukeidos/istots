from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from istots.app.doctor import (
    DoctorArgumentError,
    DoctorResult,
    DoctorRequest,
    execute_doctor_plan,
    plan_doctor_request,
)
from istots.ocr import PaddleOCRVLRuntimeOverrides, Qwen35RuntimeOverrides


def test_plan_doctor_request_rejects_missing_category() -> None:
    with pytest.raises(DoctorArgumentError) as excinfo:
        plan_doctor_request(DoctorRequest())

    assert "doctor requires a category" in str(excinfo.value)


def test_plan_doctor_request_requires_workflow_input_sup() -> None:
    with pytest.raises(DoctorArgumentError) as excinfo:
        plan_doctor_request(
            DoctorRequest(
                category="workflow",
                target="default",
            )
        )

    assert "doctor workflow requires --input-sup" in str(excinfo.value)


def test_plan_doctor_request_builds_runtime_overrides_and_resolves_workflow_input(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")

    plan = plan_doctor_request(
        DoctorRequest(
            category="workflow",
            target="wider",
            input_sup=input_sup,
            paddle_profile="cpu",
            paddle_port=19001,
            qwen_profile="cpu",
            qwen_port=19002,
            use_temp_ocr_image_files=False,
        )
    )

    assert plan.category == "workflow"
    assert plan.target == "wider"
    assert plan.input_sup == input_sup.resolve()
    assert plan.use_temp_ocr_image_files is False
    assert plan.paddle_overrides == PaddleOCRVLRuntimeOverrides(
        profile="cpu",
        port=19001,
        threads=None,
        threads_batch=None,
        gpu_layers=None,
        no_mmproj_offload=None,
        startup_timeout_sec=120.0,
        ctx_size=None,
    )
    assert plan.qwen_overrides == Qwen35RuntimeOverrides(
        profile="cpu",
        port=19002,
        threads=None,
        threads_batch=None,
        gpu_layers=None,
        no_mmproj_offload=None,
        startup_timeout_sec=120.0,
        ctx_size=None,
        n_predict=None,
        reasoning=None,
    )


def test_execute_doctor_plan_dispatches_runtime_paddle(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "istots.app.doctor.doctor_module.run_paddle_runtime_doctor",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(ok=True, checks=(), category="runtime", target="paddle"),
    )

    plan = plan_doctor_request(
        DoctorRequest(
            category="runtime",
            target="paddle",
            models_dir=tmp_path,
            paddle_profile="cpu",
            paddle_port=19001,
        )
    )
    result = execute_doctor_plan(plan)

    assert isinstance(result, DoctorResult)
    assert result.category == "runtime"
    assert captured["models_dir"] == tmp_path
    assert captured["overrides"].profile == "cpu"
    assert captured["overrides"].port == 19001


def test_execute_doctor_plan_dispatches_runtime_qwen(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"

    monkeypatch.setattr(
        "istots.app.doctor.doctor_module.run_qwen_runtime_doctor",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(ok=True, checks=(), category="runtime", target="qwen"),
    )

    plan = plan_doctor_request(
        DoctorRequest(
            category="runtime",
            target="qwen",
            corrector_model_path=model_path,
            corrector_mmproj_path=mmproj_path,
            qwen_profile="cpu",
        )
    )
    execute_doctor_plan(plan)

    assert captured["explicit_model_path"] == model_path
    assert captured["explicit_mmproj_path"] == mmproj_path
    assert captured["overrides"].profile == "cpu"


def test_execute_doctor_plan_dispatches_workflow(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")

    monkeypatch.setattr(
        "istots.app.doctor.doctor_module.run_workflow_doctor",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(ok=True, checks=(), category="workflow", target="default"),
    )

    plan = plan_doctor_request(
        DoctorRequest(
            category="workflow",
            target="default",
            input_sup=input_sup,
            use_temp_ocr_image_files=False,
        )
    )
    execute_doctor_plan(plan)

    assert captured["workflow"] == "default"
    assert captured["input_sup"] == input_sup.resolve()
    assert captured["use_temp_ocr_image_files"] is False
