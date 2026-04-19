from __future__ import annotations

from pathlib import Path

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
