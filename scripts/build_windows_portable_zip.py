from __future__ import annotations

import sys
from pathlib import Path

from istots.windows_build import (
    build_windows_portable_archive,
    verify_windows_portable_inputs,
    windows_portable_build_layout,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    project_root = _project_root()
    layout = windows_portable_build_layout(project_root)

    try:
        verify_windows_portable_inputs(layout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Build the Windows GUI bundle first with `uv run python scripts/build_windows_gui.py`.",
            file=sys.stderr,
        )
        return 1

    output_path = build_windows_portable_archive(layout)
    print(f"Built Windows portable archive: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
