from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from istots import doctor, llama_runtime


def test_report_to_check_accepts_string_profile() -> None:
    report = llama_runtime.LlamaServerDoctorReport(
        role=llama_runtime.LlamaServerRole.OCR,
        profile="auto",
        launch_spec=None,
        issues=tuple(),
    )

    check = doctor._report_to_check("runtime:ocr", report)  # noqa: SLF001

    assert check.ok is True
    assert dict(check.details) == {
        "role": "ocr",
        "profile": "auto",
    }


def test_report_to_check_exposes_bind_and_connect_hosts(tmp_path: Path) -> None:
    report = llama_runtime.LlamaServerDoctorReport(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        launch_spec=llama_runtime.LlamaServerLaunchSpec(
            role=llama_runtime.LlamaServerRole.OCR,
            profile=llama_runtime.LlamaServerProfile.AUTO,
            binary_path=tmp_path / "llama-server",
            model_path=tmp_path / "model.gguf",
            mmproj_path=tmp_path / "mmproj.gguf",
            host="0.0.0.0",
            port=18080,
        ),
        issues=tuple(),
    )

    check = doctor._report_to_check("runtime:ocr", report)  # noqa: SLF001

    assert dict(check.details)["bind_host"] == "0.0.0.0"
    assert dict(check.details)["connect_host"] == "127.0.0.1"


def test_run_llama_server_doctor_normalizes_string_profile_on_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(llama_runtime, "detect_llama_server_path", lambda explicit=None: None)

    report = llama_runtime.run_llama_server_doctor(
        role=llama_runtime.LlamaServerRole.OCR,
        overrides=llama_runtime.LlamaServerOverrides(profile="cpu"),
    )

    assert report.profile is llama_runtime.LlamaServerProfile.CPU
    assert report.launch_spec is None
    assert report.issues[0].code == "missing_binary"


def test_build_llama_server_launch_spec_normalizes_string_profile(monkeypatch, tmp_path: Path) -> None:
    gguf_dir = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    monkeypatch.setattr(llama_runtime, "resolve_local_model_path", lambda model_id, models_dir=None: gguf_dir)

    spec = llama_runtime.build_llama_server_launch_spec(
        role=llama_runtime.LlamaServerRole.OCR,
        binary_path=tmp_path / "llama-server",
        models_dir=tmp_path,
        overrides=llama_runtime.LlamaServerOverrides(profile="cpu", ctx_size=3072),
    )

    assert spec.profile is llama_runtime.LlamaServerProfile.CPU
    assert spec.ctx_size == 3072


def test_run_workflow_smoke_check_cleans_temp_artifacts_on_success(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "doctor-workflow"

    def fake_mkdtemp(*, prefix: str) -> str:
        output_dir.mkdir()
        return str(output_dir)

    def fake_convert_sup_to_srt(**kwargs):
        kwargs["output_srt"].write_text("1\n", encoding="utf-8")
        kwargs["detector_output"].write_text("[]\n", encoding="utf-8")
        return SimpleNamespace(
            output_srt=kwargs["output_srt"],
            processed_count=2,
            written_count=2,
            detector_record_count=0,
            correction_record_count=0,
            correction_applied_count=0,
        )

    monkeypatch.setattr(doctor.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(doctor, "convert_sup_to_srt", fake_convert_sup_to_srt)

    check = doctor._run_workflow_smoke_check(  # noqa: SLF001
        input_sup=tmp_path / "input.sup",
        models_dir=None,
        detector_mode="default",
        explicit_binary_path=None,
        host="127.0.0.1",
        min_pixels=32768,
        paddle_overrides=doctor.PaddleOCRVLRuntimeOverrides(),
        corrector_config=None,
        use_temp_ocr_image_files=True,
    )

    assert check.ok is True
    assert dict(check.details) == {
        "processed_count": "2",
        "written_count": "2",
        "detector_record_count": "0",
        "temp_artifacts": "cleaned",
    }
    assert output_dir.exists() is False


def test_run_workflow_smoke_check_keeps_temp_artifacts_on_failure(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "doctor-workflow"

    def fake_mkdtemp(*, prefix: str) -> str:
        output_dir.mkdir()
        return str(output_dir)

    def fake_convert_sup_to_srt(**kwargs):
        kwargs["output_srt"].write_text("partial\n", encoding="utf-8")
        kwargs["detector_output"].write_text("[]\n", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(doctor.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(doctor, "convert_sup_to_srt", fake_convert_sup_to_srt)

    check = doctor._run_workflow_smoke_check(  # noqa: SLF001
        input_sup=tmp_path / "input.sup",
        models_dir=None,
        detector_mode="default",
        explicit_binary_path=None,
        host="127.0.0.1",
        min_pixels=32768,
        paddle_overrides=doctor.PaddleOCRVLRuntimeOverrides(),
        corrector_config=None,
        use_temp_ocr_image_files=True,
    )

    assert check.ok is False
    assert check.issues[0].code == "workflow_smoke_failed"
    assert "retained temporary workflow artifacts" in check.issues[0].message
    assert dict(check.details) == {
        "artifact_dir": str(output_dir),
    }
    assert output_dir.is_dir()
    assert (output_dir / "doctor.srt").read_text(encoding="utf-8") == "partial\n"
    assert (output_dir / "doctor.detector.jsonl").read_text(encoding="utf-8") == "[]\n"


def test_run_gemini_auth_doctor_reports_invalid_auth_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bad-auth.json"
    config_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setenv("ISTOTS_AUTH_CONFIG_PATH", str(config_path))

    result = doctor.run_gemini_auth_doctor()

    assert result.ok is False
    assert len(result.checks) == 1
    check = result.checks[0]
    assert check.name == "auth:gemini"
    assert check.issues[0].code == "invalid_config"
    assert "invalid Gemini auth config" in check.issues[0].message
    assert str(config_path) in check.issues[0].message
    assert "Expecting property name enclosed in double quotes" not in check.issues[0].message
    assert dict(check.details) == {"config_path": str(config_path)}


def test_run_workflow_doctor_skips_smoke_when_gemini_auth_check_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    smoke_called = False

    def fake_run_gemini_auth_doctor(*, api_key_env: str = "GEMINI_API_KEY"):
        return doctor.DoctorSuiteResult(
            category="auth",
            target="gemini",
            checks=(
                doctor.DoctorCheckResult(
                    name="auth:gemini",
                    ok=False,
                    issues=(
                        llama_runtime.LlamaServerDoctorIssue(
                            code="invalid_config",
                            message="invalid Gemini auth config at /tmp/auth.json. Fix or remove the file, then rerun.",
                        ),
                    ),
                    details=(("config_path", "/tmp/auth.json"),),
                ),
            ),
        )

    def fake_run_workflow_smoke_check(**kwargs):
        nonlocal smoke_called
        smoke_called = True
        return doctor.DoctorCheckResult(name="workflow:smoke", ok=True)

    monkeypatch.setattr(doctor, "run_gemini_auth_doctor", fake_run_gemini_auth_doctor)
    monkeypatch.setattr(doctor, "_run_workflow_smoke_check", fake_run_workflow_smoke_check)

    result = doctor.run_workflow_doctor(
        workflow="corrector-gemini",
        input_sup=input_sup,
    )

    assert smoke_called is False
    assert [check.name for check in result.checks] == ["workflow:input-sup", "auth:gemini"]
    assert result.checks[1].ok is False
