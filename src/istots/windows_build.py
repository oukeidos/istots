from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

WINDOWS_GUI_APP_NAME = "istots"
WINDOWS_GUI_DIST_DIRNAME = "windows-gui"
WINDOWS_GUI_WORK_RELATIVE_PATH = Path("build") / "pyinstaller" / WINDOWS_GUI_DIST_DIRNAME
WINDOWS_GUI_DIST_RELATIVE_PATH = Path("dist") / WINDOWS_GUI_DIST_DIRNAME
WINDOWS_GUI_SPEC_RELATIVE_PATH = Path("packaging") / "pyinstaller" / "istots_gui.spec"
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
