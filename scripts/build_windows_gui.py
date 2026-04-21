from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from istots.windows_build import (
    build_windows_gui_command,
    missing_packaged_documents,
    stage_windows_gui_bundle_assets,
    verify_windows_gui_bundle,
    windows_gui_build_layout,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def main() -> int:
    if sys.platform != "win32":
        print("Windows GUI packaging is supported only on Windows.", file=sys.stderr)
        return 1

    project_root = _project_root()
    missing_documents = missing_packaged_documents(project_root)
    if missing_documents:
        print("The Windows GUI bundle is missing required attached documents:", file=sys.stderr)
        for path in missing_documents:
            print(f"- {path}", file=sys.stderr)
        return 1

    layout = windows_gui_build_layout(project_root)
    if not layout.spec_path.exists():
        print(f"PyInstaller spec file is missing: {layout.spec_path}", file=sys.stderr)
        return 1

    command = build_windows_gui_command(layout, python_executable=Path(sys.executable))
    print(f"Running Windows GUI build: {_format_command(command)}")
    try:
        subprocess.run(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1

    stage_windows_gui_bundle_assets(layout)

    try:
        verify_windows_gui_bundle(layout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Built Windows GUI bundle: {layout.bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
