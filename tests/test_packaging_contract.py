from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_keeps_hf_runtime_optional() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = set(data["project"]["dependencies"])
    optional_hf = set(data["project"]["optional-dependencies"]["hf"])

    assert "torch>=2.6.0" not in dependencies
    assert "transformers>=5.0.0" not in dependencies
    assert "torch>=2.6.0" in optional_hf
    assert "transformers>=5.0.0" in optional_hf


def test_pyproject_uses_release_gguf_package() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = set(data["project"]["dependencies"])

    assert "gguf==0.18.0" in dependencies
    assert not any("git+https://github.com/ggml-org/llama.cpp" in dep for dep in dependencies)
