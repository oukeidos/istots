from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from istots import gguf_support


def test_ensure_known_good_gguf_py_downloads_from_local_source(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    snapshot_root = tmp_path / "support"

    files = {
        "LICENSE": b"mit\n",
        "gguf/__init__.py": b"from .constants import VALUE\n",
        "gguf/constants.py": b"VALUE = 1\n",
    }
    for relative_path, data in files.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    manifest = {
        relative_path: hashlib.sha256(data).hexdigest()
        for relative_path, data in files.items()
    }
    monkeypatch.setattr(gguf_support, "GGUF_PY_FILES", manifest)
    monkeypatch.setattr(gguf_support, "GGUF_PY_COMMIT", "test-commit")

    snapshot_dir = gguf_support.ensure_known_good_gguf_py(
        support_dir=snapshot_root,
        base_url=str(source_root),
    )

    assert snapshot_dir == (snapshot_root / "gguf-py" / "test-commit").resolve()
    for relative_path, data in files.items():
        assert (snapshot_dir / relative_path).read_bytes() == data
    assert (snapshot_dir / gguf_support.MANIFEST_NAME).exists()


def test_load_known_good_gguf_prefers_installed_package(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(gguf_support, "get_installed_gguf", lambda: sentinel)
    module = gguf_support.load_known_good_gguf(source_mode="installed")
    assert module is sentinel


def test_load_known_good_gguf_installed_mode_rejects_missing(monkeypatch) -> None:
    monkeypatch.setattr(gguf_support, "get_installed_gguf", lambda: None)
    with pytest.raises(RuntimeError, match="no installed gguf package is available"):
        gguf_support.load_known_good_gguf(source_mode="installed")


def test_load_known_good_gguf_auto_mode_falls_back_to_snapshot(monkeypatch) -> None:
    sentinel = object()
    calls: list[tuple[Path | None, str | None]] = []

    monkeypatch.setattr(gguf_support, "get_installed_gguf", lambda: None)
    monkeypatch.setattr(
        gguf_support,
        "ensure_known_good_gguf_py",
        lambda support_dir=None, base_url=None, force=False: (
            calls.append((support_dir, base_url)) or Path("/tmp/snapshot")
        ),
    )
    monkeypatch.setattr(gguf_support, "import_known_good_gguf", lambda snapshot_dir: sentinel)

    module = gguf_support.load_known_good_gguf(
        source_mode="auto",
        support_dir=Path("/tmp/support"),
        base_url="https://example.invalid/gguf-py",
    )

    assert module is sentinel
    assert calls == [(Path("/tmp/support"), "https://example.invalid/gguf-py")]
