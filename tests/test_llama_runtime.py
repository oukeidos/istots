from __future__ import annotations

from pathlib import Path

from istots import llama_runtime


def test_build_llama_server_command_uses_auto_profile_defaults() -> None:
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18080,
    )

    assert llama_runtime.build_llama_server_command(spec) == [
        "/tmp/llama-server",
        "-m",
        "/tmp/model.gguf",
        "--mmproj",
        "/tmp/mmproj.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
    ]


def test_build_llama_server_command_applies_cpu_profile_and_overrides() -> None:
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.CPU,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18080,
        threads=12,
        threads_batch=8,
    )

    assert llama_runtime.build_llama_server_command(spec) == [
        "/tmp/llama-server",
        "-m",
        "/tmp/model.gguf",
        "--mmproj",
        "/tmp/mmproj.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "-t",
        "12",
        "-tb",
        "8",
        "--no-mmproj-offload",
    ]


def test_resolve_llama_server_role_assets_uses_derived_mmproj_for_fast_role(monkeypatch, tmp_path: Path) -> None:
    gguf_dir = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    monkeypatch.setattr(llama_runtime, "resolve_local_model_path", lambda model_id, models_dir=None: gguf_dir)

    assets = llama_runtime.resolve_llama_server_role_assets(
        llama_runtime.LlamaServerRole.OCR_FAST,
        models_dir=tmp_path,
        min_pixels=32768,
    )

    assert assets.model_path == (gguf_dir / "PaddleOCR-VL-1.5.gguf").resolve()
    assert assets.mmproj_path == (gguf_dir / "PaddleOCR-VL-1.5-mmproj.minpix32768.gguf").resolve()


def test_run_llama_server_doctor_reports_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(llama_runtime, "detect_llama_server_path", lambda explicit=None: None)
    report = llama_runtime.run_llama_server_doctor(role="ocr")
    assert report.ok is False
    assert report.issues[0].code == "missing_binary"


def test_run_llama_server_doctor_runs_smoke_on_ready_runtime(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    process = object()
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")

    monkeypatch.setattr(llama_runtime, "detect_llama_server_path", lambda explicit=None: binary)
    monkeypatch.setattr(
        llama_runtime,
        "build_llama_server_launch_spec",
        lambda **kwargs: llama_runtime.LlamaServerLaunchSpec(
            role=llama_runtime.LlamaServerRole.OCR,
            profile=llama_runtime.LlamaServerProfile.AUTO,
            binary_path=binary,
            model_path=model,
            mmproj_path=mmproj,
            host="127.0.0.1",
            port=18080,
        ),
    )
    monkeypatch.setattr(llama_runtime, "is_port_in_use", lambda host, port: False)
    monkeypatch.setattr(llama_runtime, "start_llama_server", lambda spec, startup_timeout_sec: process)
    monkeypatch.setattr(llama_runtime, "request_llama_server_smoke", lambda spec: "OK")

    stopped: list[object] = []
    monkeypatch.setattr(llama_runtime, "stop_llama_server", lambda proc: stopped.append(proc))

    report = llama_runtime.run_llama_server_doctor(role="ocr")

    assert report.ok is True
    assert report.smoke_response == "OK"
    assert stopped == [process]
