from __future__ import annotations

import argparse
import sys
from pathlib import Path

from istots.gui.qt_app import (
    MissingGuiDependencyError,
    launch_gui,
    list_gui_theme_ids,
    render_theme_previews,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="istots-gui")
    parser.add_argument(
        "--theme",
        choices=list_gui_theme_ids(),
        default="warm",
        help="Select a design preview theme for the GUI shell.",
    )
    parser.add_argument(
        "--render-theme-previews",
        metavar="DIR",
        type=Path,
        help="Render preview screenshots for all GUI themes into DIR and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.render_theme_previews is not None:
            for path in render_theme_previews(args.render_theme_previews):
                print(path)
            return 0
        return launch_gui(theme_id=args.theme)
    except MissingGuiDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
