from __future__ import annotations

from pathlib import Path

import pytest

from istots.windows_build import (
    PACKAGED_DOCUMENT_NAMES,
    build_windows_gui_command,
    packaged_document_paths,
    stage_windows_gui_bundle_assets,
    verify_windows_gui_bundle,
    windows_gui_build_layout,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_packaged_document_paths_match_fixed_policy() -> None:
    project_root = _project_root()

    document_paths = packaged_document_paths(project_root)

    assert tuple(path.name for path in document_paths) == PACKAGED_DOCUMENT_NAMES
    assert "AGENTS.md" not in {path.name for path in document_paths}
    assert "TASK.md" not in {path.name for path in document_paths}
    for path in document_paths:
        assert path.exists()


def test_windows_gui_build_layout_uses_expected_paths() -> None:
    project_root = _project_root()

    layout = windows_gui_build_layout(project_root)

    assert layout.spec_path == project_root / "packaging" / "pyinstaller" / "istots_gui.spec"
    assert layout.dist_root == project_root / "dist" / "windows-gui"
    assert layout.work_root == project_root / "build" / "pyinstaller" / "windows-gui"
    assert layout.bundle_root == project_root / "dist" / "windows-gui" / "istots"
    assert layout.docs_dir == layout.bundle_root / "docs"
    assert layout.executable_path == layout.bundle_root / "istots.exe"
    assert layout.bundle_app_icon_path == layout.bundle_root / "istots.ico"
    assert layout.app_icon_source_path.exists()
    assert layout.installer_icon_source_path.exists()


def test_build_windows_gui_command_uses_pyinstaller_module() -> None:
    layout = windows_gui_build_layout(_project_root())

    command = build_windows_gui_command(layout, python_executable=Path("C:/Python311/python.exe"))

    assert command[:3] == ("C:\\Python311\\python.exe", "-m", "PyInstaller")
    assert "--distpath" in command
    assert "--workpath" in command
    assert str(layout.spec_path) == command[-1]


def test_verify_windows_gui_bundle_requires_executable_icon_and_documents(tmp_path: Path) -> None:
    layout = windows_gui_build_layout(tmp_path)

    layout.bundle_root.mkdir(parents=True, exist_ok=True)
    layout.docs_dir.mkdir(parents=True, exist_ok=True)
    layout.executable_path.write_text("", encoding="utf-8")
    layout.bundle_app_icon_path.write_text("", encoding="utf-8")
    for name in PACKAGED_DOCUMENT_NAMES:
        (layout.docs_dir / name).write_text(name, encoding="utf-8")

    verify_windows_gui_bundle(layout)


def test_verify_windows_gui_bundle_reports_missing_files(tmp_path: Path) -> None:
    layout = windows_gui_build_layout(tmp_path)
    layout.bundle_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="Windows GUI bundle verification failed"):
        verify_windows_gui_bundle(layout)


def test_stage_windows_gui_bundle_assets_copies_documents_and_icon(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    layout = windows_gui_build_layout(project_root)

    layout.app_icon_source_path.parent.mkdir(parents=True, exist_ok=True)
    layout.app_icon_source_path.write_text("icon", encoding="utf-8")
    layout.installer_icon_source_path.parent.mkdir(parents=True, exist_ok=True)
    layout.installer_icon_source_path.write_text("setup-icon", encoding="utf-8")
    for source_path in packaged_document_paths(project_root):
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source_path.name, encoding="utf-8")

    stage_windows_gui_bundle_assets(layout)

    assert layout.bundle_app_icon_path.read_text(encoding="utf-8") == "icon"
    for name in PACKAGED_DOCUMENT_NAMES:
        assert (layout.docs_dir / name).read_text(encoding="utf-8") == name
