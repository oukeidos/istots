from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from istots import __version__, cli
from istots import model_store, pipeline
from istots.corrector import CorrectorMode
from istots.ocr import PaddleOCRVLRuntimeOverrides, Qwen35RuntimeOverrides


def test_normalize_argv_keeps_subcommand() -> None:
    assert cli._normalize_argv(["setup"]) == ["setup"]  # noqa: SLF001


def test_normalize_argv_legacy_convert() -> None:
    assert cli._normalize_argv(["input.sup", "output.srt"]) == [  # noqa: SLF001
        "convert",
        "input.sup",
        "output.srt",
    ]


def test_run_version(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.run(["--version"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"istots {__version__}"


def test_run_help_includes_subcommand_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.run(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Subcommand Details:" in captured.out
    assert "--engine {llama-server,hf}" in captured.out
    assert "--ocr-mode {default,fast}" in captured.out
    assert "--furigana-mask" in captured.out
    assert "--detector-output DETECTOR_OUTPUT" in captured.out
    assert "--detector-mode {default,wider}" in captured.out
    assert "--detector-family-addon" in captured.out
    assert "--corrector {off,qwen-local,gemini}" in captured.out
    assert "--corrector-output CORRECTOR_OUTPUT" in captured.out
    assert "--srt-policy {safe,overlap}" in captured.out
    assert "--hf-device {auto,cpu,gpu}" in captured.out
    assert "--hf-dtype {auto,float32,float16,bfloat16}" in captured.out
    assert "--min-pixels MIN_PIXELS" in captured.out
    assert "--paddle-profile {auto,cpu}" in captured.out
    assert "--qwen-profile {auto,cpu}" in captured.out
    assert "--output-dir OUTPUT_DIR" in captured.out
    assert "--no-detector" in captured.out
    assert "--force" in captured.out
    assert "{runtime,auth,workflow}" in captured.out
    assert "--input-sup INPUT_SUP" in captured.out


def test_run_routes_setup(monkeypatch) -> None:
    def fake_setup(args) -> int:
        assert args.command == "setup"
        assert args.model_id == "abc/def"
        return 11

    monkeypatch.setattr(cli, "run_setup", fake_setup)
    assert cli.run(["setup", "--model-id", "abc/def"]) == 11


def test_run_routes_materialize_mmproj(monkeypatch) -> None:
    def fake_materialize(args) -> int:
        assert args.command == "materialize-mmproj"
        assert args.base_mmproj == Path("base.gguf")
        assert args.min_pixels == 32768
        assert args.gguf_source_mode == "auto"
        return 17

    monkeypatch.setattr(cli, "run_materialize_mmproj", fake_materialize)
    assert cli.run(["materialize-mmproj", "base.gguf"]) == 17


def test_run_routes_doctor(monkeypatch) -> None:
    def fake_doctor(args) -> int:
        assert args.command == "doctor"
        assert args.doctor_category == "runtime"
        assert args.doctor_target == "paddle"
        return 19

    monkeypatch.setattr(cli, "run_doctor", fake_doctor)
    assert cli.run(["doctor", "runtime", "paddle"]) == 19


def test_run_routes_smoke(monkeypatch) -> None:
    def fake_smoke(args) -> int:
        assert args.command == "smoke"
        assert args.ocr_mode == "default"
        return 23

    monkeypatch.setattr(cli, "run_smoke", fake_smoke)
    assert cli.run(["smoke"]) == 23


def test_run_routes_auth(monkeypatch) -> None:
    def fake_auth(args) -> int:
        assert args.command == "auth"
        assert args.auth_provider == "gemini"
        return 29

    monkeypatch.setattr(cli, "run_auth", fake_auth)
    assert cli.run(["auth", "gemini", "status"]) == 29


def test_run_setup_downloads_hf_and_gguf_assets(monkeypatch, tmp_path: Path) -> None:
    artifacts = SimpleNamespace(
        hf_model_dir=tmp_path / "hf_model",
        gguf_model_dir=tmp_path / "gguf_model",
        gguf_model_path=tmp_path / "gguf_model" / "PaddleOCR-VL-1.5.gguf",
        gguf_mmproj_path=tmp_path / "gguf_model" / "PaddleOCR-VL-1.5-mmproj.gguf",
        gguf_mmproj_minpix32768_path=tmp_path / "gguf_model" / "PaddleOCR-VL-1.5-mmproj.minpix32768.gguf",
        qwen_corrector_dir=None,
        qwen_corrector_model_path=None,
        qwen_corrector_mmproj_path=None,
    )
    captured: dict[str, object] = {}

    def fake_setup_default_runtime_assets(**kwargs):
        captured.update(kwargs)
        return artifacts

    monkeypatch.setattr(
        model_store,
        "setup_default_runtime_assets",
        fake_setup_default_runtime_assets,
    )

    rc = cli.run(
        [
            "setup",
            "--model-id",
            "hf/model",
            "--gguf-model-id",
            "gguf/model",
            "--models-dir",
            str(tmp_path),
            "--gguf-source-mode",
            "installed",
            "--min-pixels",
            "32768",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["hf_model_id"] == "hf/model"
    assert captured["gguf_model_id"] == "gguf/model"
    assert captured["with_qwen_corrector"] is False
    assert captured["models_dir"] == tmp_path
    assert captured["gguf_source_mode"] == "installed"
    assert captured["min_pixels"] == 32768


def test_run_setup_can_enable_qwen_corrector_download(monkeypatch, tmp_path: Path) -> None:
    artifacts = SimpleNamespace(
        hf_model_dir=tmp_path / "hf_model",
        gguf_model_dir=tmp_path / "gguf_model",
        gguf_model_path=tmp_path / "gguf_model" / "PaddleOCR-VL-1.5.gguf",
        gguf_mmproj_path=tmp_path / "gguf_model" / "PaddleOCR-VL-1.5-mmproj.gguf",
        gguf_mmproj_minpix32768_path=tmp_path / "gguf_model" / "PaddleOCR-VL-1.5-mmproj.minpix32768.gguf",
        qwen_corrector_dir=tmp_path / "qwen_model",
        qwen_corrector_model_path=tmp_path / "qwen_model" / "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        qwen_corrector_mmproj_path=tmp_path / "qwen_model" / "mmproj-BF16.gguf",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        model_store,
        "setup_default_runtime_assets",
        lambda **kwargs: captured.update(kwargs) or artifacts,
    )

    rc = cli.run(
        [
            "setup",
            "--with-qwen-corrector",
            "--qwen-corrector-model-id",
            "unsloth/Qwen3.5-35B-A3B-GGUF",
            "--qwen-corrector-model-filename",
            "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            "--qwen-corrector-mmproj-filename",
            "mmproj-BF16.gguf",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["with_qwen_corrector"] is True
    assert captured["qwen_corrector_model_id"] == "unsloth/Qwen3.5-35B-A3B-GGUF"
    assert captured["qwen_corrector_model_filename"] == "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    assert captured["qwen_corrector_mmproj_filename"] == "mmproj-BF16.gguf"


def test_run_setup_rejects_existing_local_model_id(tmp_path: Path) -> None:
    local_dir = tmp_path / "existing-model-dir"
    local_dir.mkdir()

    rc = cli.run(["setup", "--model-id", str(local_dir), "--quiet"])

    assert rc == 1


def test_run_materialize_mmproj_applies_requested_value(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    output = tmp_path / "derived.gguf"

    def fake_materialize_mmproj(**kwargs):
        captured["materialize"] = kwargs
        return output

    def fake_read_mmproj_min_pixels(*args, **kwargs):
        captured["read"] = {"args": args, "kwargs": kwargs}
        return 32768

    monkeypatch.setattr(cli, "configure_logging", lambda verbose: None)
    monkeypatch.setattr("istots.llama_mmproj.materialize_mmproj", fake_materialize_mmproj)
    monkeypatch.setattr("istots.llama_mmproj.read_mmproj_min_pixels", fake_read_mmproj_min_pixels)

    rc = cli.run(
        [
            "materialize-mmproj",
            "base.gguf",
            "--output",
            str(output),
            "--min-pixels",
            "32768",
            "--gguf-source-mode",
            "installed",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["materialize"]["base_mmproj"] == Path("base.gguf")
    assert captured["materialize"]["output_path"] == output
    assert captured["materialize"]["gguf_source_mode"] == "installed"
    assert captured["read"]["args"] == (output,)


def test_run_doctor_runtime_paddle_passes_family_overrides(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_paddle_runtime_doctor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(category="runtime", target="paddle", ok=True, checks=())

    monkeypatch.setattr("istots.doctor.run_paddle_runtime_doctor", fake_run_paddle_runtime_doctor)

    rc = cli.run(
        [
            "doctor",
            "runtime",
            "paddle",
            "--models-dir",
            str(tmp_path),
            "--paddle-profile",
            "cpu",
            "--paddle-port",
            "19001",
            "--paddle-threads",
            "12",
            "--paddle-threads-batch",
            "8",
            "--paddle-no-mmproj-offload",
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["models_dir"] == tmp_path
    overrides = captured["overrides"]
    assert overrides.profile == "cpu"
    assert overrides.port == 19001
    assert overrides.threads == 12
    assert overrides.threads_batch == 8
    assert overrides.no_mmproj_offload is True


def test_run_doctor_runtime_qwen_passes_family_overrides(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_qwen_runtime_doctor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(category="runtime", target="qwen", ok=True, checks=())

    monkeypatch.setattr("istots.doctor.run_qwen_runtime_doctor", fake_run_qwen_runtime_doctor)

    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    rc = cli.run(
        [
            "doctor",
            "runtime",
            "qwen",
            "--corrector-model-path",
            str(model_path),
            "--corrector-mmproj-path",
            str(mmproj_path),
            "--qwen-profile",
            "cpu",
            "--qwen-no-mmproj-offload",
            "--quiet",
        ]
    )

    assert rc == 0
    overrides = captured["overrides"]
    assert overrides.profile == "cpu"
    assert overrides.no_mmproj_offload is True
    assert captured["explicit_model_path"] == model_path
    assert captured["explicit_mmproj_path"] == mmproj_path


def test_run_doctor_auth_gemini_routes_api_key_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_gemini_auth_doctor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(category="auth", target="gemini", ok=True, checks=())

    monkeypatch.setattr("istots.doctor.run_gemini_auth_doctor", fake_run_gemini_auth_doctor)

    rc = cli.run(["doctor", "auth", "gemini", "--api-key-env", "GOOGLE_API_KEY", "--quiet"])

    assert rc == 0
    assert captured["api_key_env"] == "GOOGLE_API_KEY"


def test_run_doctor_workflow_requires_explicit_input_sup(monkeypatch) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda verbose: None)

    with pytest.raises(SystemExit):
        cli.run(["doctor", "workflow", "default", "--quiet"])


def test_run_doctor_workflow_passes_input_sup(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_workflow_doctor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(category="workflow", target="wider", ok=True, checks=())

    monkeypatch.setattr("istots.doctor.run_workflow_doctor", fake_run_workflow_doctor)

    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    rc = cli.run(
        [
            "doctor",
            "workflow",
            "wider",
            "--input-sup",
            str(input_sup),
            "--quiet",
        ]
    )

    assert rc == 0
    assert captured["workflow"] == "wider"
    assert captured["input_sup"] == input_sup.resolve()


def test_run_routes_legacy_convert(monkeypatch) -> None:
    def fake_convert(args) -> int:
        assert args.command == "convert"
        assert args.input_sup == Path("input.sup")
        assert args.output_srt == Path("output.srt")
        return 0

    monkeypatch.setattr(cli, "run_convert", fake_convert)
    assert cli.run(["input.sup", "output.srt"]) == 0


def test_run_convert_uses_local_model_and_offline(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_ensure_local_model(model_id: str, models_dir: Path | None = None) -> Path:
        assert model_id == "org/model"
        assert models_dir is None
        return model_dir

    captured: dict[str, object] = {}

    def fake_convert_sup_to_srt(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            written_count=0,
            output_srt=Path("output.srt"),
            device_used="cpu",
        )

    monkeypatch.setattr(model_store, "ensure_local_model", fake_ensure_local_model)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    output_srt = tmp_path / "output.srt"
    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--model-id", "org/model"])
    assert rc == 0
    assert captured["engine"] == "hf"
    assert captured["model_id"] == str(model_dir)
    assert captured["local_files_only"] is True
    assert captured["enable_furigana_mask"] is False
    assert captured["srt_policy"] == "safe"
    assert captured["input_sup"] == input_sup.resolve()
    assert captured["output_srt"] == output_srt.resolve()


def test_run_convert_passes_hf_device_and_dtype(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: model_dir)
    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs)
        or SimpleNamespace(
            written_count=0,
            output_srt=tmp_path / "output.srt",
            device_used="gpu",
        ),
    )

    output_srt = tmp_path / "output.srt"
    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--engine",
            "hf",
            "--hf-device",
            "gpu",
            "--hf-dtype",
            "bfloat16",
        ]
    )

    assert rc == 0
    assert captured["hf_device"] == "gpu"
    assert captured["hf_dtype"] == "bfloat16"


def test_run_convert_passes_furigana_mask_flag(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_convert_sup_to_srt(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            written_count=0,
            output_srt=tmp_path / "output.srt",
            device_used="cpu",
        )

    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: model_dir)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    output_srt = tmp_path / "output.srt"
    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--furigana-mask"])
    assert rc == 0
    assert captured["enable_furigana_mask"] is True


def test_run_convert_passes_srt_policy(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_convert_sup_to_srt(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            written_count=0,
            output_srt=tmp_path / "output.srt",
            device_used="cpu",
        )

    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: model_dir)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    output_srt = tmp_path / "output.srt"
    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--srt-policy", "overlap"])
    assert rc == 0
    assert captured["srt_policy"] == "overlap"


def test_run_convert_defaults_to_llama_server(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"

    captured: dict[str, object] = {}
    called = False

    def fake_convert_sup_to_srt(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="gpu",
        )

    def fake_ensure_local_model(model_id, models_dir=None):
        nonlocal called
        called = True
        return tmp_path / "unused"

    monkeypatch.setattr(model_store, "ensure_local_model", fake_ensure_local_model)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    rc = cli.run([str(input_sup), str(output_srt), "--quiet"])

    assert rc == 0
    assert called is False
    assert captured["engine"] == "llama-server"
    assert captured["ocr_mode"] == "default"
    assert captured["local_files_only"] is False
    assert captured["models_dir"] is None


def test_run_smoke_uses_default_sample_and_auto_detector(monkeypatch, tmp_path: Path) -> None:
    sample_sup = tmp_path / "sample.sup"
    sample_sup.write_bytes(b"PG")
    output_dir = tmp_path / "smoke"
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_default_smoke_input_sup", lambda: sample_sup.resolve())
    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_dir / "sample.smoke.srt",
            device_used="cpu",
            detector_record_count=0,
            correction_record_count=0,
            correction_applied_count=0,
        ),
    )

    rc = cli.run(["smoke", "--output-dir", str(output_dir), "--quiet"])

    assert rc == 0
    assert captured["input_sup"] == sample_sup.resolve()
    assert captured["output_srt"] == (output_dir / "sample.smoke.srt").resolve()
    assert captured["detector_output"] == (output_dir / "sample.detector.jsonl").resolve()
    assert captured["engine"] == "llama-server"
    assert captured["ocr_mode"] == "default"
    assert captured["models_dir"] is None


def test_run_smoke_disables_detector_for_fast_mode(monkeypatch, tmp_path: Path) -> None:
    sample_sup = tmp_path / "sample.sup"
    sample_sup.write_bytes(b"PG")
    output_dir = tmp_path / "smoke"
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_default_smoke_input_sup", lambda: sample_sup.resolve())
    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_dir / "sample.smoke.srt",
            device_used="cpu",
        ),
    )

    rc = cli.run(["smoke", "--output-dir", str(output_dir), "--ocr-mode", "fast", "--quiet"])

    assert rc == 0
    assert captured["ocr_mode"] == "fast"
    assert captured["detector_output"] is None


def test_run_smoke_auto_writes_corrector_manifest(monkeypatch, tmp_path: Path) -> None:
    sample_sup = tmp_path / "sample.sup"
    sample_sup.write_bytes(b"PG")
    output_dir = tmp_path / "smoke"
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "qwen-mmproj.gguf"
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_default_smoke_input_sup", lambda: sample_sup.resolve())
    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_dir / "sample.smoke.srt",
            device_used="cpu",
            detector_record_count=0,
            correction_record_count=1,
            correction_applied_count=1,
        ),
    )

    rc = cli.run(
        [
            "smoke",
            "--output-dir",
            str(output_dir),
            "--quiet",
            "--corrector",
            "qwen-local",
            "--corrector-model-path",
            str(model_path),
            "--corrector-mmproj-path",
            str(mmproj_path),
        ]
    )

    assert rc == 0
    config = captured["corrector_config"]
    assert config.mode is CorrectorMode.QWEN_LOCAL
    assert config.output_path == (output_dir / "sample.corrected.jsonl").resolve()


def test_run_convert_passes_llama_runtime_overrides(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--paddle-profile",
            "cpu",
            "--paddle-port",
            "19005",
            "--paddle-threads",
            "12",
            "--paddle-threads-batch",
            "8",
            "--paddle-gpu-layers",
            "0",
            "--paddle-no-mmproj-offload",
            "--paddle-startup-timeout-sec",
            "30",
        ]
    )

    assert rc == 0
    assert captured["engine"] == "llama-server"
    assert captured["ocr_mode"] == "default"
    assert captured["paddle_runtime_overrides"] == PaddleOCRVLRuntimeOverrides(
        profile="cpu",
        port=19005,
        threads=12,
        threads_batch=8,
        gpu_layers=0,
        no_mmproj_offload=True,
        startup_timeout_sec=30.0,
    )


def test_run_convert_passes_fast_ocr_mode(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--ocr-mode",
            "fast",
            "--paddle-profile",
            "cpu",
        ]
    )

    assert rc == 0
    assert captured["engine"] == "llama-server"
    assert captured["ocr_mode"] == "fast"
    assert captured["paddle_runtime_overrides"] == PaddleOCRVLRuntimeOverrides(profile="cpu")


def test_run_convert_passes_detector_output(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
            detector_record_count=2,
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--detector-output",
            str(detector_output),
        ]
    )

    assert rc == 0
    assert captured["detector_output"] == detector_output


def test_run_convert_passes_detector_mode(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
            detector_record_count=2,
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--detector-output",
            str(detector_output),
            "--detector-mode",
            "wider",
        ]
    )

    assert rc == 0
    assert captured["detector_mode"] == "wider"


def test_run_convert_passes_detector_family_addon(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
            detector_record_count=3,
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--detector-output",
            str(detector_output),
            "--detector-family-addon",
        ]
    )

    assert rc == 0
    assert captured["detector_family_addon"] is True


def test_run_convert_passes_qwen_local_corrector_config(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    corrector_output = tmp_path / "corrected.jsonl"
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "qwen-mmproj.gguf"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--corrector",
            "qwen-local",
            "--corrector-model-path",
            str(model_path),
            "--corrector-mmproj-path",
            str(mmproj_path),
            "--corrector-output",
            str(corrector_output),
            "--qwen-port",
            "19083",
        ]
    )

    assert rc == 0
    config = captured["corrector_config"]
    assert config.mode is CorrectorMode.QWEN_LOCAL
    assert config.local_model_path == model_path.resolve()
    assert config.local_mmproj_path == mmproj_path.resolve()
    assert config.output_path == corrector_output.resolve()
    assert config.local_runtime_overrides == Qwen35RuntimeOverrides(port=19083)


def test_run_convert_passes_qwen_local_mmproj_offload_override(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "qwen-mmproj.gguf"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--corrector",
            "qwen-local",
            "--corrector-model-path",
            str(model_path),
            "--corrector-mmproj-path",
            str(mmproj_path),
            "--qwen-no-mmproj-offload",
        ]
    )

    assert rc == 0
    config = captured["corrector_config"]
    assert config.mode is CorrectorMode.QWEN_LOCAL
    assert config.local_runtime_overrides == Qwen35RuntimeOverrides(no_mmproj_offload=True)


def test_run_convert_keeps_paddle_and_qwen_runtime_overrides_isolated(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "qwen-mmproj.gguf"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--paddle-profile",
            "cpu",
            "--corrector",
            "qwen-local",
            "--corrector-model-path",
            str(model_path),
            "--corrector-mmproj-path",
            str(mmproj_path),
            "--qwen-port",
            "19083",
            "--qwen-no-mmproj-offload",
        ]
    )

    assert rc == 0
    assert captured["paddle_runtime_overrides"] == PaddleOCRVLRuntimeOverrides(profile="cpu")
    config = captured["corrector_config"]
    assert config.local_runtime_overrides == Qwen35RuntimeOverrides(
        port=19083,
        no_mmproj_offload=True,
    )


def test_run_convert_resolves_default_qwen_local_corrector_assets(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    captured: dict[str, object] = {}
    resolved_model_path = tmp_path / "downloaded-qwen.gguf"
    resolved_mmproj_path = tmp_path / "downloaded-qwen-mmproj.gguf"

    monkeypatch.setattr(
        model_store,
        "ensure_local_qwen_corrector_assets",
        lambda models_dir=None: (resolved_model_path, resolved_mmproj_path),
    )
    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--corrector",
            "qwen-local",
        ]
    )

    assert rc == 0
    config = captured["corrector_config"]
    assert config.mode is CorrectorMode.QWEN_LOCAL
    assert config.local_model_path == resolved_model_path
    assert config.local_mmproj_path == resolved_mmproj_path
    assert config.local_runtime_overrides == Qwen35RuntimeOverrides()


def test_run_convert_passes_gemini_corrector_config(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    cache_dir = tmp_path / "cache"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--corrector",
            "gemini",
            "--corrector-gemini-model",
            "gemini-test",
            "--corrector-thinking-level",
            "low",
            "--corrector-cache-dir",
            str(cache_dir),
        ]
    )

    assert rc == 0
    config = captured["corrector_config"]
    assert config.mode is CorrectorMode.GEMINI
    assert config.gemini_model == "gemini-test"
    assert config.thinking_level == "low"
    assert config.cache_dir == cache_dir.resolve()


def test_run_convert_existing_output_noninteractive_requires_force(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    output_srt.write_text("existing", encoding="utf-8")

    called = False

    def fake_convert_sup_to_srt(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        )

    monkeypatch.setattr(cli, "_can_prompt_for_overwrite", lambda: False)
    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: tmp_path)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--model-id", "org/model"])
    assert rc == 1
    assert called is False


def test_run_convert_existing_output_force_overwrites(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    output_srt.write_text("existing", encoding="utf-8")
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_convert_sup_to_srt(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt.resolve(),
            device_used="cpu",
        )

    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: model_dir)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--model-id", "org/model", "--force"])
    assert rc == 0
    assert captured["output_srt"] == output_srt.resolve()


def test_run_convert_existing_output_prompt_yes(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    output_srt.write_text("existing", encoding="utf-8")
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    called = False

    def fake_convert_sup_to_srt(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt.resolve(),
            device_used="cpu",
        )

    monkeypatch.setattr(cli, "_can_prompt_for_overwrite", lambda: True)
    monkeypatch.setattr(cli, "_confirm_overwrite", lambda path: True)
    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: model_dir)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--model-id", "org/model"])
    assert rc == 0
    assert called is True


def test_run_convert_existing_output_prompt_no_cancels(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    output_srt.write_text("existing", encoding="utf-8")

    called = False

    def fake_convert_sup_to_srt(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt.resolve(),
            device_used="cpu",
        )

    monkeypatch.setattr(cli, "_can_prompt_for_overwrite", lambda: True)
    monkeypatch.setattr(cli, "_confirm_overwrite", lambda path: False)
    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: tmp_path)
    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--model-id", "org/model"])
    assert rc == 1
    assert called is False


def test_run_convert_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    input_sup = tmp_path / "same.sup"
    input_sup.write_bytes(b"PG")

    with pytest.raises(SystemExit) as excinfo:
        cli.run([str(input_sup), str(input_sup), "--quiet"])

    assert excinfo.value.code == 2


def test_run_convert_rejects_same_input_and_detector_output_path(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    input_sup = tmp_path / "input.sup"
    original_bytes = b"PG-input"
    input_sup.write_bytes(original_bytes)
    output_srt = tmp_path / "output.srt"

    called = False

    def fake_convert_sup_to_srt(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt.resolve(),
            device_used="cpu",
        )

    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--detector-output",
                str(input_sup),
            ]
        )

    assert excinfo.value.code == 2
    assert called is False
    assert input_sup.read_bytes() == original_bytes
    captured = capsys.readouterr()
    assert "input_sup and detector_output must be different paths" in captured.err


def test_run_convert_rejects_same_input_and_corrector_output_path(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    input_sup = tmp_path / "input.sup"
    original_bytes = b"PG-input"
    input_sup.write_bytes(original_bytes)
    output_srt = tmp_path / "output.srt"
    model_path = tmp_path / "qwen.gguf"
    mmproj_path = tmp_path / "qwen-mmproj.gguf"

    called = False

    def fake_convert_sup_to_srt(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt.resolve(),
            device_used="cpu",
        )

    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--corrector",
                "qwen-local",
                "--corrector-model-path",
                str(model_path),
                "--corrector-mmproj-path",
                str(mmproj_path),
                "--corrector-output",
                str(input_sup),
            ]
        )

    assert excinfo.value.code == 2
    assert called is False
    assert input_sup.read_bytes() == original_bytes
    captured = capsys.readouterr()
    assert "input_sup and corrector_output must be different paths" in captured.err


def test_run_convert_rejects_same_output_and_detector_output_path(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    input_sup = tmp_path / "input.sup"
    original_bytes = b"PG-input"
    input_sup.write_bytes(original_bytes)
    output_srt = tmp_path / "output.srt"

    called = False

    def fake_convert_sup_to_srt(**kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(
            written_count=0,
            output_srt=output_srt.resolve(),
            device_used="cpu",
        )

    monkeypatch.setattr(pipeline, "convert_sup_to_srt", fake_convert_sup_to_srt)

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--detector-output",
                str(output_srt),
            ]
        )

    assert excinfo.value.code == 2
    assert called is False
    assert input_sup.read_bytes() == original_bytes
    captured = capsys.readouterr()
    assert "output_srt and detector_output must be different paths" in captured.err


def test_run_convert_rejects_output_directory(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_dir = tmp_path / "outdir"
    output_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        cli.run([str(input_sup), str(output_dir), "--quiet"])

    assert excinfo.value.code == 2


def test_run_convert_allows_fast_mode_for_hf(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    model_dir = tmp_path / "cached_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(model_store, "ensure_local_model", lambda model_id, models_dir=None: model_dir)
    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--ocr-mode", "fast"])

    assert rc == 0
    assert captured["engine"] == "hf"
    assert captured["ocr_mode"] == "fast"


def test_run_convert_accepts_shared_paddle_port_for_fast_mode(monkeypatch, tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "convert_sup_to_srt",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
            written_count=0,
            output_srt=output_srt,
            device_used="cpu",
        ),
    )

    rc = cli.run(
        [
            str(input_sup),
            str(output_srt),
            "--quiet",
            "--ocr-mode",
            "fast",
            "--paddle-port",
            "19005",
        ]
    )

    assert rc == 0
    assert captured["ocr_mode"] == "fast"
    assert captured["paddle_runtime_overrides"] == PaddleOCRVLRuntimeOverrides(port=19005)


def test_run_convert_rejects_detector_output_for_hf(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--engine",
                "hf",
                "--detector-output",
                str(detector_output),
            ]
        )

    assert excinfo.value.code == 2


def test_run_convert_rejects_detector_output_for_fast_mode(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    detector_output = tmp_path / "detector.jsonl"

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--ocr-mode",
                "fast",
                "--detector-output",
                str(detector_output),
            ]
    )

    assert excinfo.value.code == 2


def test_run_convert_rejects_detector_family_addon_without_detector_or_corrector(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--detector-family-addon",
            ]
        )

    assert excinfo.value.code == 2


def test_run_convert_rejects_detector_mode_without_detector_or_corrector(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--detector-mode",
                "wider",
            ]
        )

    assert excinfo.value.code == 2


def test_run_convert_rejects_corrector_for_hf(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"

    with pytest.raises(SystemExit) as excinfo:
        cli.run([str(input_sup), str(output_srt), "--quiet", "--engine", "hf", "--corrector", "gemini"])

    assert excinfo.value.code == 2


def test_run_convert_rejects_qwen_local_without_paths(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--corrector",
                "qwen-local",
                "--corrector-model-path",
                str(tmp_path / "qwen.gguf"),
            ]
        )

    assert excinfo.value.code == 2


def test_run_convert_rejects_qwen_mmproj_offload_override_without_qwen_local(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"

    with pytest.raises(SystemExit) as excinfo:
        cli.run(
            [
                str(input_sup),
                str(output_srt),
                "--quiet",
                "--qwen-no-mmproj-offload",
            ]
        )

    assert excinfo.value.code == 2


def test_run_auth_status_prints_source_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "istots.gemini_auth.get_gemini_auth_status",
        lambda: SimpleNamespace(
            keyring_backend="test.backend",
            keyring_configured=True,
            env_file_configured=True,
            env_file_path=Path("/tmp/test.env"),
            env_file_contains_key=False,
            process_env_configured=False,
            process_env_name=None,
            effective_source="keyring",
        ),
    )

    rc = cli.run(["auth", "gemini", "status"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "keyring: configured (test.backend)" in captured.out
    assert ".env path: configured (/tmp/test.env)" in captured.out
    assert "shell env: missing" in captured.out
    assert "effective source: keyring" in captured.out


def test_run_auth_env_file_set(monkeypatch, capsys, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(
        "istots.gemini_auth.set_configured_gemini_env_file",
        lambda path: path.resolve(),
    )

    rc = cli.run(["auth", "gemini", "env-file", "set", str(env_path)])

    assert rc == 0
    captured = capsys.readouterr()
    assert f"Configured Gemini .env file: {env_path.resolve()}" in captured.out


def test_run_auth_set_uses_hidden_prompt(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "secret-key")
    monkeypatch.setattr("istots.gemini_auth.set_gemini_api_key", lambda api_key: "test.backend")

    rc = cli.run(["auth", "gemini", "set"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "Gemini API key stored in keyring backend: test.backend" in captured.out


def test_run_convert_rejects_corrector_output_without_corrector(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_srt = tmp_path / "output.srt"
    corrector_output = tmp_path / "corrected.jsonl"

    with pytest.raises(SystemExit) as excinfo:
        cli.run([str(input_sup), str(output_srt), "--quiet", "--corrector-output", str(corrector_output)])

    assert excinfo.value.code == 2
