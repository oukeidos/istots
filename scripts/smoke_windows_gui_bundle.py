from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from istots.windows_build import verify_windows_gui_bundle, windows_gui_build_layout


REQUIRED_SMOKE_OUTPUT_NAMES = (
    "warm.png",
    "theme_compare_sheet.png",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    if sys.platform != "win32":
        print("Windows GUI smoke testing is supported only on Windows.", file=sys.stderr)
        return 1

    project_root = _project_root()
    layout = windows_gui_build_layout(project_root)
    try:
        verify_windows_gui_bundle(layout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_dir = (project_root / "build" / "windows-gui-smoke").resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = (
        str(layout.executable_path),
        "--render-theme-previews",
        str(output_dir),
    )
    print(f"Running packaged GUI smoke test: {layout.executable_path}")
    try:
        subprocess.run(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1

    missing_outputs = [name for name in REQUIRED_SMOKE_OUTPUT_NAMES if not (output_dir / name).exists()]
    if missing_outputs:
        print("Windows GUI smoke test did not produce the expected outputs:", file=sys.stderr)
        for name in missing_outputs:
            print(f"- {output_dir / name}", file=sys.stderr)
        return 1

    print(f"Packaged GUI smoke outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
