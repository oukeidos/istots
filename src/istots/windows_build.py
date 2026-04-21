from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tomllib
from typing import Iterable

WINDOWS_GUI_APP_NAME = "istots"
WINDOWS_INSTALLER_APP_NAME = "IStoTS"
WINDOWS_INSTALLER_APP_PUBLISHER = "oukeidos"
WINDOWS_INSTALLER_APP_GUID = "07ac00d9-1e18-4ee9-8af6-01c007408576"
WINDOWS_INSTALLER_APP_ID = "{{" + WINDOWS_INSTALLER_APP_GUID + "}"
WINDOWS_GUI_DIST_DIRNAME = "windows-gui"
WINDOWS_GUI_WORK_RELATIVE_PATH = Path("build") / "pyinstaller" / WINDOWS_GUI_DIST_DIRNAME
WINDOWS_GUI_DIST_RELATIVE_PATH = Path("dist") / WINDOWS_GUI_DIST_DIRNAME
WINDOWS_RELEASE_RELATIVE_PATH = Path("dist") / "windows-release"
WINDOWS_GUI_SPEC_RELATIVE_PATH = Path("packaging") / "pyinstaller" / "istots_gui.spec"
WINDOWS_INNO_RELATIVE_PATH = Path("packaging") / "inno"
WINDOWS_INNO_SCRIPT_RELATIVE_PATH = WINDOWS_INNO_RELATIVE_PATH / "istots_gui.iss"
WINDOWS_INNO_OUTPUT_RELATIVE_PATH = WINDOWS_INNO_RELATIVE_PATH / "Output"
WINDOWS_APP_ICON_RELATIVE_PATH = (
    Path("src") / "istots" / "resources" / "icons" / "windows" / "istots.ico"
)
WINDOWS_INSTALLER_ICON_RELATIVE_PATH = (
    Path("src") / "istots" / "resources" / "icons" / "windows" / "istots_setup.ico"
)
PACKAGED_DOCUMENT_NAMES = (
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "DISCLAIMER.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
)
WINDOWS_PYTHON_OPENSSL_DLL_NAMES = ("libcrypto-3-x64.dll", "libssl-3-x64.dll")


@dataclass(frozen=True)
class WindowsGuiBuildLayout:
    project_root: Path
    spec_path: Path
    dist_root: Path
    work_root: Path
    bundle_root: Path
    docs_dir: Path
    executable_path: Path
    bundle_app_icon_path: Path
    app_icon_source_path: Path
    installer_icon_source_path: Path


@dataclass(frozen=True)
class WindowsInstallerBuildLayout:
    project_root: Path
    gui_bundle_layout: WindowsGuiBuildLayout
    script_path: Path
    output_dir: Path
    output_base_filename: str


@dataclass(frozen=True)
class WindowsPortableBuildLayout:
    project_root: Path
    gui_bundle_layout: WindowsGuiBuildLayout
    output_dir: Path
    output_base_filename: str


def packaged_document_paths(project_root: Path) -> tuple[Path, ...]:
    root = project_root.expanduser().resolve()
    return tuple((root / name).resolve() for name in PACKAGED_DOCUMENT_NAMES)


def packaged_document_datas(project_root: Path) -> list[tuple[str, str]]:
    return [(str(path), "docs") for path in packaged_document_paths(project_root)]


def missing_packaged_documents(project_root: Path) -> tuple[Path, ...]:
    return tuple(path for path in packaged_document_paths(project_root) if not path.exists())


def windows_gui_build_layout(project_root: Path) -> WindowsGuiBuildLayout:
    root = project_root.expanduser().resolve()
    dist_root = (root / WINDOWS_GUI_DIST_RELATIVE_PATH).resolve()
    bundle_root = (dist_root / WINDOWS_GUI_APP_NAME).resolve()
    docs_dir = (bundle_root / "docs").resolve()
    app_icon_source_path = (root / WINDOWS_APP_ICON_RELATIVE_PATH).resolve()
    return WindowsGuiBuildLayout(
        project_root=root,
        spec_path=(root / WINDOWS_GUI_SPEC_RELATIVE_PATH).resolve(),
        dist_root=dist_root,
        work_root=(root / WINDOWS_GUI_WORK_RELATIVE_PATH).resolve(),
        bundle_root=bundle_root,
        docs_dir=docs_dir,
        executable_path=(bundle_root / f"{WINDOWS_GUI_APP_NAME}.exe").resolve(),
        bundle_app_icon_path=(bundle_root / app_icon_source_path.name).resolve(),
        app_icon_source_path=app_icon_source_path,
        installer_icon_source_path=(root / WINDOWS_INSTALLER_ICON_RELATIVE_PATH).resolve(),
    )


def project_version(project_root: Path) -> str:
    pyproject_path = project_root.expanduser().resolve() / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def windows_installer_output_base_filename(*, version: str) -> str:
    return f"{WINDOWS_INSTALLER_APP_NAME}-{version}-windows-x64-setup"


def windows_portable_output_base_filename(*, version: str) -> str:
    return f"{WINDOWS_INSTALLER_APP_NAME}-{version}-windows-x64-portable"


def windows_installer_build_layout(project_root: Path) -> WindowsInstallerBuildLayout:
    root = project_root.expanduser().resolve()
    version = project_version(root)
    return WindowsInstallerBuildLayout(
        project_root=root,
        gui_bundle_layout=windows_gui_build_layout(root),
        script_path=(root / WINDOWS_INNO_SCRIPT_RELATIVE_PATH).resolve(),
        output_dir=(root / WINDOWS_INNO_OUTPUT_RELATIVE_PATH).resolve(),
        output_base_filename=windows_installer_output_base_filename(version=version),
    )


def windows_portable_build_layout(project_root: Path) -> WindowsPortableBuildLayout:
    root = project_root.expanduser().resolve()
    version = project_version(root)
    return WindowsPortableBuildLayout(
        project_root=root,
        gui_bundle_layout=windows_gui_build_layout(root),
        output_dir=(root / WINDOWS_RELEASE_RELATIVE_PATH).resolve(),
        output_base_filename=windows_portable_output_base_filename(version=version),
    )


def build_windows_gui_command(
    layout: WindowsGuiBuildLayout,
    *,
    python_executable: Path,
) -> tuple[str, ...]:
    return (
        str(python_executable.expanduser().resolve()),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(layout.dist_root),
        "--workpath",
        str(layout.work_root),
        str(layout.spec_path),
    )


def build_windows_installer_command(
    layout: WindowsInstallerBuildLayout,
    *,
    compiler_path: Path,
) -> tuple[str, ...]:
    return (
        str(compiler_path.expanduser().resolve()),
        f"/DMyAppVersion={project_version(layout.project_root)}",
        f"/DMyBundleRoot={layout.gui_bundle_layout.bundle_root}",
        f"/DMyOutputDir={layout.output_dir}",
        f"/DMyOutputBaseFilename={layout.output_base_filename}",
        str(layout.script_path),
    )


def expected_windows_installer_output_path(layout: WindowsInstallerBuildLayout) -> Path:
    return (layout.output_dir / f"{layout.output_base_filename}.exe").resolve()


def expected_windows_portable_archive_path(layout: WindowsPortableBuildLayout) -> Path:
    return (layout.output_dir / f"{layout.output_base_filename}.zip").resolve()


def verify_windows_installer_inputs(layout: WindowsInstallerBuildLayout) -> None:
    verify_windows_gui_bundle(layout.gui_bundle_layout)
    if not layout.script_path.exists():
        raise RuntimeError(f"Windows installer script is missing: {layout.script_path}")


def verify_windows_portable_inputs(layout: WindowsPortableBuildLayout) -> None:
    verify_windows_gui_bundle(layout.gui_bundle_layout)


def inno_setup_compiler_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured_path = os.environ.get("ISCC_EXE")
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_name)
        if not base:
            continue
        candidates.append(Path(base) / "Inno Setup 6" / "ISCC.exe")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return tuple(deduped)


def detect_inno_setup_compiler() -> Path | None:
    for candidate in inno_setup_compiler_candidates():
        if candidate.exists():
            return candidate
    which_path = shutil.which("ISCC.exe")
    if which_path:
        return Path(which_path).expanduser().resolve()
    return None


def required_bundle_paths(layout: WindowsGuiBuildLayout) -> tuple[Path, ...]:
    required_paths = [layout.executable_path, layout.bundle_app_icon_path]
    required_paths.extend((layout.docs_dir / name).resolve() for name in PACKAGED_DOCUMENT_NAMES)
    return tuple(required_paths)


def verify_windows_gui_bundle(layout: WindowsGuiBuildLayout) -> None:
    missing = [path for path in required_bundle_paths(layout) if not path.exists()]
    if not missing:
        return
    missing_lines = "\n".join(f"- {path}" for path in missing)
    raise RuntimeError(
        "Windows GUI bundle verification failed. The following packaged files are missing:\n"
        f"{missing_lines}"
    )


def stage_windows_gui_bundle_assets(layout: WindowsGuiBuildLayout) -> None:
    layout.bundle_root.mkdir(parents=True, exist_ok=True)
    layout.docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(layout.app_icon_source_path, layout.bundle_app_icon_path)
    for source_path in packaged_document_paths(layout.project_root):
        shutil.copy2(source_path, layout.docs_dir / source_path.name)


def build_windows_portable_archive(layout: WindowsPortableBuildLayout) -> Path:
    verify_windows_portable_inputs(layout)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = expected_windows_portable_archive_path(layout)
    if output_path.exists():
        output_path.unlink()
    archive_base = str(output_path.with_suffix(""))
    created_archive = shutil.make_archive(
        archive_base,
        "zip",
        root_dir=layout.gui_bundle_layout.dist_root,
        base_dir=layout.gui_bundle_layout.bundle_root.name,
    )
    return Path(created_archive).resolve()


def python_runtime_dll_dir(*, base_prefix: Path | None = None) -> Path:
    runtime_root = Path(sys.base_prefix if base_prefix is None else base_prefix).expanduser().resolve()
    return (runtime_root / "DLLs").resolve()


def python_runtime_openssl_paths(*, base_prefix: Path | None = None) -> tuple[Path, ...]:
    dll_dir = python_runtime_dll_dir(base_prefix=base_prefix)
    paths = tuple((dll_dir / name).resolve() for name in WINDOWS_PYTHON_OPENSSL_DLL_NAMES)
    missing = tuple(path for path in paths if not path.exists())
    if not missing:
        return paths
    missing_lines = "\n".join(f"- {path}" for path in missing)
    raise RuntimeError(
        "Python runtime OpenSSL DLL discovery failed. The following files are missing:\n"
        f"{missing_lines}"
    )


def python_runtime_openssl_binary_specs(*, base_prefix: Path | None = None) -> list[tuple[str, str]]:
    return [(str(path), ".") for path in python_runtime_openssl_paths(base_prefix=base_prefix)]


def pyinstaller_binary_toc_entries(
    binary_specs: Iterable[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    toc_entries: list[tuple[str, str, str]] = []
    for source_path_text, target_dir in binary_specs:
        source_path = Path(source_path_text).expanduser().resolve()
        normalized_target = "." if target_dir in ("", ".") else target_dir.replace("\\", "/").strip("/")
        dest_name = source_path.name if normalized_target == "." else f"{normalized_target}/{source_path.name}"
        toc_entries.append((dest_name, str(source_path), "BINARY"))
    return toc_entries


def filter_pyinstaller_binaries_by_name(
    binaries: Iterable[tuple[str, str, str]],
    *,
    excluded_names: Iterable[str],
) -> list[tuple[str, str, str]]:
    excluded = {name.casefold() for name in excluded_names}
    filtered: list[tuple[str, str, str]] = []
    for entry in binaries:
        if Path(str(entry[0])).name.casefold() in excluded:
            continue
        filtered.append(entry)
    return filtered
