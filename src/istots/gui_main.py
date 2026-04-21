from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

from istots import llama_runtime
from istots.runtime_diagnostics import append_runtime_diagnostic_event, install_faulthandler_trace
from istots.runtime_stdio import ensure_standard_streams

ensure_standard_streams()
install_faulthandler_trace()

if __name__ == "__main__":
    multiprocessing.freeze_support()

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
    ensure_standard_streams()
    install_faulthandler_trace()
    multiprocessing.freeze_support()
    args = _build_parser().parse_args(argv)
    append_runtime_diagnostic_event(
        "gui_main_start",
        argv=argv or [],
        theme=args.theme,
        render_theme_previews=args.render_theme_previews,
    )
    try:
        if args.render_theme_previews is not None:
            append_runtime_diagnostic_event(
                "gui_preview_render_start",
                output_dir=args.render_theme_previews,
            )
            for path in render_theme_previews(args.render_theme_previews):
                print(path)
            append_runtime_diagnostic_event(
                "gui_preview_render_complete",
                output_dir=args.render_theme_previews,
            )
            return 0
        append_runtime_diagnostic_event("gui_runtime_cleanup_start")
        llama_runtime.clear_llama_server_process_shutdown_request()
        llama_runtime.cleanup_stale_managed_llama_server_processes()
        append_runtime_diagnostic_event("gui_launch_start", theme_id=args.theme)
        exit_code = launch_gui(theme_id=args.theme)
        append_runtime_diagnostic_event("gui_launch_exit", exit_code=exit_code)
        return exit_code
    except MissingGuiDependencyError as exc:
        append_runtime_diagnostic_event(
            "gui_missing_dependency",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
