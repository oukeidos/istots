from __future__ import annotations

from pathlib import Path
import tomllib

from istots.resources import icon_bundle_root


def test_pyproject_keeps_hf_runtime_optional() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = set(data["project"]["dependencies"])
    optional_hf = set(data["project"]["optional-dependencies"]["hf"])

    assert "torch>=2.6.0" not in dependencies
    assert "transformers>=5.0.0" not in dependencies
    assert "torch>=2.6.0" in optional_hf
    assert "transformers>=5.0.0" in optional_hf


def test_pyproject_keeps_gui_runtime_optional() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = set(data["project"]["dependencies"])
    optional_gui = set(data["project"]["optional-dependencies"]["gui"])

    assert not any(dep.startswith("PySide6") for dep in dependencies)
    assert "PySide6>=6.8.0" in optional_gui


def test_pyproject_uses_release_gguf_package() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = set(data["project"]["dependencies"])

    assert "gguf==0.18.0" in dependencies
    assert not any("git+https://github.com/ggml-org/llama.cpp" in dep for dep in dependencies)


def test_packaged_gui_icon_bundle_keeps_cross_platform_outputs() -> None:
    bundle = icon_bundle_root()

    assert bundle.joinpath("README.md").is_file()
    assert bundle.joinpath("png", "generic", "istots_256.png").is_file()
    assert bundle.joinpath("windows", "istots.ico").is_file()
    assert bundle.joinpath("windows", "istots_setup.ico").is_file()
    assert bundle.joinpath("macos", "istots.icns").is_file()
    assert bundle.joinpath("linux", "hicolor", "scalable", "apps", "istots.svg").is_file()
