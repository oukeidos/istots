from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from istots import __version__, cli
from istots import model_store, pipeline


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
    assert "--batch-size BATCH_SIZE" in captured.out
    assert "--force" in captured.out


def test_run_routes_setup(monkeypatch) -> None:
    def fake_setup(args) -> int:
        assert args.command == "setup"
        assert args.model_id == "abc/def"
        return 11

    monkeypatch.setattr(cli, "run_setup", fake_setup)
    assert cli.run(["setup", "--model-id", "abc/def"]) == 11


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
    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--model-id", "org/model"])
    assert rc == 0
    assert captured["model_id"] == str(model_dir)
    assert captured["local_files_only"] is True
    assert captured["input_sup"] == input_sup.resolve()
    assert captured["output_srt"] == output_srt.resolve()


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

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--model-id", "org/model"])
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

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--model-id", "org/model", "--force"])
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

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--model-id", "org/model"])
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

    rc = cli.run([str(input_sup), str(output_srt), "--quiet", "--model-id", "org/model"])
    assert rc == 1
    assert called is False


def test_run_convert_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    input_sup = tmp_path / "same.sup"
    input_sup.write_bytes(b"PG")

    with pytest.raises(SystemExit) as excinfo:
        cli.run([str(input_sup), str(input_sup), "--quiet"])

    assert excinfo.value.code == 2


def test_run_convert_rejects_output_directory(tmp_path: Path) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_dir = tmp_path / "outdir"
    output_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        cli.run([str(input_sup), str(output_dir), "--quiet"])

    assert excinfo.value.code == 2
