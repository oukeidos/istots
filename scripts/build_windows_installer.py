from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from istots.windows_build import (
    build_windows_installer_command,
    detect_inno_setup_compiler,
    expected_windows_installer_output_path,
    verify_windows_installer_inputs,
    windows_installer_build_layout,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def main() -> int:
    if sys.platform != "win32":
        print("Windows installer authoring is supported only on Windows.", file=sys.stderr)
        return 1

    project_root = _project_root()
    layout = windows_installer_build_layout(project_root)

    try:
        verify_windows_installer_inputs(layout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Build the Windows GUI bundle first with `uv run python scripts/build_windows_gui.py`.",
            file=sys.stderr,
        )
        return 1

    compiler_path = detect_inno_setup_compiler()
    if compiler_path is None:
        print(
            "Inno Setup compiler was not found. Install Inno Setup 6 or set ISCC_EXE to ISCC.exe.",
            file=sys.stderr,
        )
        return 1

    layout.output_dir.mkdir(parents=True, exist_ok=True)
    command = build_windows_installer_command(layout, compiler_path=compiler_path)
    print(f"Running Windows installer build: {_format_command(command)}")
    try:
        subprocess.run(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1

    output_path = expected_windows_installer_output_path(layout)
    if not output_path.exists():
        print(f"Windows installer build did not produce the expected output: {output_path}", file=sys.stderr)
        return 1

    print(f"Built Windows installer: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
