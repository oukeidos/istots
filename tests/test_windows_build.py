from __future__ import annotations

from pathlib import Path

import pytest

from istots.windows_build import (
    PACKAGED_DOCUMENT_NAMES,
    WINDOWS_PYTHON_OPENSSL_DLL_NAMES,
    build_windows_gui_command,
    build_windows_portable_archive,
    build_windows_installer_command,
    detect_inno_setup_compiler,
    expected_windows_portable_archive_path,
    expected_windows_installer_output_path,
    filter_pyinstaller_binaries_by_name,
    inno_setup_compiler_candidates,
    packaged_document_paths,
    pyinstaller_binary_toc_entries,
    python_runtime_dll_dir,
    python_runtime_openssl_binary_specs,
    python_runtime_openssl_paths,
    project_version,
    stage_windows_gui_bundle_assets,
    verify_windows_gui_bundle,
    verify_windows_portable_inputs,
    verify_windows_installer_inputs,
    windows_gui_build_layout,
    windows_portable_build_layout,
    windows_portable_output_base_filename,
    windows_installer_build_layout,
    windows_installer_output_base_filename,
    WINDOWS_INSTALLER_APP_GUID,
    WINDOWS_INSTALLER_APP_ID,
    WINDOWS_INSTALLER_APP_NAME,
    WINDOWS_INSTALLER_APP_PUBLISHER,
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


def test_windows_installer_metadata_is_fixed() -> None:
    assert WINDOWS_INSTALLER_APP_NAME == "IStoTS"
    assert WINDOWS_INSTALLER_APP_PUBLISHER == "oukeidos"
    assert WINDOWS_INSTALLER_APP_GUID == "07ac00d9-1e18-4ee9-8af6-01c007408576"
    assert WINDOWS_INSTALLER_APP_ID == "{{07ac00d9-1e18-4ee9-8af6-01c007408576}"


def test_python_runtime_dll_dir_uses_base_prefix_dlls_subdir(tmp_path: Path) -> None:
    runtime_root = tmp_path / "python-runtime"

    dll_dir = python_runtime_dll_dir(base_prefix=runtime_root)

    assert dll_dir == runtime_root / "DLLs"


def test_python_runtime_openssl_paths_read_from_python_dlls_dir(tmp_path: Path) -> None:
    runtime_root = tmp_path / "python-runtime"
    dll_dir = runtime_root / "DLLs"
    dll_dir.mkdir(parents=True, exist_ok=True)

    expected_paths = []
    for name in WINDOWS_PYTHON_OPENSSL_DLL_NAMES:
        path = dll_dir / name
        path.write_text(name, encoding="utf-8")
        expected_paths.append(path)

    assert python_runtime_openssl_paths(base_prefix=runtime_root) == tuple(expected_paths)
    assert python_runtime_openssl_binary_specs(base_prefix=runtime_root) == [
        (str(path), ".") for path in expected_paths
    ]


def test_python_runtime_openssl_paths_fail_when_runtime_dll_is_missing(tmp_path: Path) -> None:
    runtime_root = tmp_path / "python-runtime"
    dll_dir = runtime_root / "DLLs"
    dll_dir.mkdir(parents=True, exist_ok=True)
    (dll_dir / WINDOWS_PYTHON_OPENSSL_DLL_NAMES[0]).write_text("openssl", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Python runtime OpenSSL DLL discovery failed"):
        python_runtime_openssl_paths(base_prefix=runtime_root)


def test_pyinstaller_binary_toc_entries_map_dot_target_to_bundle_root(tmp_path: Path) -> None:
    source_path = tmp_path / "DLLs" / WINDOWS_PYTHON_OPENSSL_DLL_NAMES[0]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("openssl", encoding="utf-8")

    toc_entries = pyinstaller_binary_toc_entries([(str(source_path), ".")])

    assert toc_entries == [(source_path.name, str(source_path), "BINARY")]


def test_filter_pyinstaller_binaries_by_name_removes_path_leaked_openssl_entries() -> None:
    binaries = [
        ("libssl-3-x64.dll", r"C:\msys64\mingw64\bin\libssl-3-x64.dll", "BINARY"),
        ("subdir/LIBCRYPTO-3-X64.DLL", r"C:\git\bin\libcrypto-3-x64.dll", "BINARY"),
        ("VCRUNTIME140.dll", r"C:\Python\DLLs\VCRUNTIME140.dll", "BINARY"),
    ]

    filtered = filter_pyinstaller_binaries_by_name(
        binaries,
        excluded_names=WINDOWS_PYTHON_OPENSSL_DLL_NAMES,
    )

    assert filtered == [("VCRUNTIME140.dll", r"C:\Python\DLLs\VCRUNTIME140.dll", "BINARY")]


def test_project_version_reads_the_pyproject_version() -> None:
    assert project_version(_project_root()) == "0.4.7"


def test_windows_installer_build_layout_uses_expected_paths() -> None:
    project_root = _project_root()
    version = project_version(project_root)

    layout = windows_installer_build_layout(project_root)

    assert layout.gui_bundle_layout == windows_gui_build_layout(project_root)
    assert layout.script_path == project_root / "packaging" / "inno" / "istots_gui.iss"
    assert layout.output_dir == project_root / "packaging" / "inno" / "Output"
    assert layout.output_base_filename == f"IStoTS-{version}-windows-x64-setup"


def test_windows_portable_build_layout_uses_expected_paths() -> None:
    project_root = _project_root()
    version = project_version(project_root)

    layout = windows_portable_build_layout(project_root)

    assert layout.gui_bundle_layout == windows_gui_build_layout(project_root)
    assert layout.output_dir == project_root / "dist" / "windows-release"
    assert layout.output_base_filename == f"IStoTS-{version}-windows-x64-portable"


def test_inno_setup_script_preserves_desktop_shortcut_choice_and_defaults_checked() -> None:
    script_text = windows_installer_build_layout(_project_root()).script_path.read_text(encoding="utf-8")

    assert "UsePreviousTasks=yes" in script_text
    assert 'Name: "desktopicon"; Description: "Create a desktop shortcut"' in script_text
    assert "Flags: unchecked" not in script_text.split("[Tasks]", 1)[1].split("[Files]", 1)[0]


def test_inno_setup_script_only_removes_managed_assets_on_manual_uninstall() -> None:
    script_text = windows_installer_build_layout(_project_root()).script_path.read_text(encoding="utf-8")

    assert "if UninstallSilent() then" in script_text
    assert "ShouldRemoveManagedAssets := False;" in script_text
    assert "Choose No to keep them for a later reinstall or upgrade." in script_text
    assert "DelTree(ManagedAssetsDir(), True, True, True)" in script_text
    assert "ExpandConstant('{localappdata}\\istots\\managed')" in script_text


def test_build_windows_gui_command_uses_pyinstaller_module() -> None:
    layout = windows_gui_build_layout(_project_root())

    command = build_windows_gui_command(layout, python_executable=Path("C:/Python311/python.exe"))

    assert command[:3] == ("C:\\Python311\\python.exe", "-m", "PyInstaller")
    assert "--distpath" in command
    assert "--workpath" in command
    assert str(layout.spec_path) == command[-1]


def test_build_windows_installer_command_uses_iscc_defines() -> None:
    layout = windows_installer_build_layout(_project_root())

    command = build_windows_installer_command(layout, compiler_path=Path("C:/Inno Setup 6/ISCC.exe"))

    assert command[0] == "C:\\Inno Setup 6\\ISCC.exe"
    assert f"/DMyAppVersion={project_version(_project_root())}" in command
    assert f"/DMyBundleRoot={layout.gui_bundle_layout.bundle_root}" in command
    assert f"/DMyOutputDir={layout.output_dir}" in command
    assert f"/DMyOutputBaseFilename={layout.output_base_filename}" in command
    assert command[-1] == str(layout.script_path)


def test_expected_windows_installer_output_path_uses_fixed_filename() -> None:
    layout = windows_installer_build_layout(_project_root())

    assert expected_windows_installer_output_path(layout) == (
        _project_root()
        / "packaging"
        / "inno"
        / "Output"
        / f"IStoTS-{project_version(_project_root())}-windows-x64-setup.exe"
    )


def test_expected_windows_portable_archive_path_uses_fixed_filename() -> None:
    layout = windows_portable_build_layout(_project_root())

    assert expected_windows_portable_archive_path(layout) == (
        _project_root()
        / "dist"
        / "windows-release"
        / f"IStoTS-{project_version(_project_root())}-windows-x64-portable.zip"
    )


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


def test_verify_windows_installer_inputs_requires_bundle_and_script(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text('[project]\nversion = "0.4.3"\n', encoding="utf-8")
    layout = windows_installer_build_layout(project_root)

    layout.gui_bundle_layout.bundle_root.mkdir(parents=True, exist_ok=True)
    layout.gui_bundle_layout.docs_dir.mkdir(parents=True, exist_ok=True)
    layout.gui_bundle_layout.executable_path.write_text("", encoding="utf-8")
    layout.gui_bundle_layout.bundle_app_icon_path.write_text("", encoding="utf-8")
    for name in PACKAGED_DOCUMENT_NAMES:
        (layout.gui_bundle_layout.docs_dir / name).write_text(name, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Windows installer script is missing"):
        verify_windows_installer_inputs(layout)


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


def test_build_windows_portable_archive_zips_the_bundle(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text('[project]\nversion = "0.4.3"\n', encoding="utf-8")
    gui_layout = windows_gui_build_layout(project_root)
    portable_layout = windows_portable_build_layout(project_root)

    gui_layout.bundle_root.mkdir(parents=True, exist_ok=True)
    gui_layout.docs_dir.mkdir(parents=True, exist_ok=True)
    gui_layout.executable_path.write_text("exe", encoding="utf-8")
    gui_layout.bundle_app_icon_path.write_text("ico", encoding="utf-8")
    for name in PACKAGED_DOCUMENT_NAMES:
        (gui_layout.docs_dir / name).write_text(name, encoding="utf-8")

    output_path = build_windows_portable_archive(portable_layout)

    assert output_path == expected_windows_portable_archive_path(portable_layout)
    assert output_path.exists()


def test_verify_windows_portable_inputs_requires_bundle(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.4.3"\n', encoding="utf-8")
    layout = windows_portable_build_layout(tmp_path)

    with pytest.raises(RuntimeError, match="Windows GUI bundle verification failed"):
        verify_windows_portable_inputs(layout)


def test_windows_installer_output_base_filename_uses_fixed_contract() -> None:
    assert windows_installer_output_base_filename(version="1.2.3") == "IStoTS-1.2.3-windows-x64-setup"


def test_windows_portable_output_base_filename_uses_fixed_contract() -> None:
    assert windows_portable_output_base_filename(version="1.2.3") == "IStoTS-1.2.3-windows-x64-portable"


def test_inno_setup_compiler_candidates_respect_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISCC_EXE", r"C:\Custom Tools\ISCC.exe")
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")

    candidates = inno_setup_compiler_candidates()

    assert candidates[0] == Path(r"C:\Custom Tools\ISCC.exe")
    assert Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe") in candidates
    assert Path(r"C:\Program Files\Inno Setup 6\ISCC.exe") in candidates


def test_detect_inno_setup_compiler_returns_none_when_no_candidate_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ISCC_EXE", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.setattr("istots.windows_build.shutil.which", lambda _binary: None)

    assert detect_inno_setup_compiler() is None


def test_windows_build_workflow_uses_supported_contract() -> None:
    workflow_path = _project_root() / ".github" / "workflows" / "windows-build.yml"

    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "runs-on: windows-2025" in workflow_text
    assert "actions/checkout@v6" in workflow_text
    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in workflow_text
    assert "actions/upload-artifact@v7" in workflow_text
    assert "uv python install 3.14.4" in workflow_text
    assert "uv sync --frozen --managed-python --python 3.14.4 --extra gui" in workflow_text
    assert (
        "uv run --managed-python --python 3.14.4 pytest --basetemp build/pytest-temp "
        "tests/test_windows_build.py"
    ) in workflow_text
    assert "uv run --managed-python --python 3.14.4 python scripts/build_windows_gui.py" in workflow_text
    assert (
        "uv run --managed-python --python 3.14.4 python scripts/smoke_windows_gui_bundle.py"
    ) in workflow_text
    assert (
        "uv run --managed-python --python 3.14.4 python scripts/build_windows_portable_zip.py"
    ) in workflow_text
    assert (
        "uv run --managed-python --python 3.14.4 python scripts/build_windows_installer.py"
    ) in workflow_text
    assert "gh release create" in workflow_text
    assert "gh release upload" in workflow_text


def test_windows_build_workflow_triggers_cover_ci_and_release() -> None:
    workflow_path = _project_root() / ".github" / "workflows" / "windows-build.yml"

    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow_text
    assert "pull_request:" in workflow_text
    assert "codex/windows-actions" in workflow_text
    assert '- "v*"' in workflow_text
