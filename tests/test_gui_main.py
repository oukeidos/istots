from __future__ import annotations

import sys
from pathlib import Path

from istots import gui_main


def test_gui_main_calls_freeze_support_before_launch(monkeypatch) -> None:
    seen: list[str] = []

    monkeypatch.setattr(gui_main, "ensure_standard_streams", lambda: seen.append("stdio"))
    monkeypatch.setattr(gui_main, "install_faulthandler_trace", lambda: seen.append("faulthandler"))
    monkeypatch.setattr(gui_main, "append_runtime_diagnostic_event", lambda *args, **kwargs: seen.append(args[0]))
    monkeypatch.setattr(gui_main.multiprocessing, "freeze_support", lambda: seen.append("freeze"))
    monkeypatch.setattr(
        gui_main.llama_runtime,
        "cleanup_stale_managed_llama_server_processes",
        lambda: seen.append("cleanup") or False,
    )
    monkeypatch.setattr(gui_main, "launch_gui", lambda theme_id: seen.append(theme_id) or 0)

    assert gui_main.main(["--theme", "poster"]) == 0
    assert seen == [
        "stdio",
        "faulthandler",
        "freeze",
        "gui_main_start",
        "gui_runtime_cleanup_start",
        "cleanup",
        "gui_launch_start",
        "poster",
        "gui_launch_exit",
    ]


def test_gui_main_recovers_when_stderr_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "__stderr__", None, raising=False)
    monkeypatch.setattr(gui_main, "install_faulthandler_trace", lambda: None)
    monkeypatch.setattr(gui_main, "append_runtime_diagnostic_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_main.multiprocessing, "freeze_support", lambda: None)
    monkeypatch.setattr(
        gui_main.llama_runtime,
        "cleanup_stale_managed_llama_server_processes",
        lambda: False,
    )
    monkeypatch.setattr(
        gui_main,
        "launch_gui",
        lambda theme_id: (_ for _ in ()).throw(gui_main.MissingGuiDependencyError("missing gui runtime")),
    )

    assert gui_main.main([]) == 1


def test_gui_main_accepts_theme_selection() -> None:
    args = gui_main._build_parser().parse_args(["--theme", "poster"])
    assert args.theme == "poster"
    assert args.render_theme_previews is None


def test_gui_main_accepts_preview_render_directory() -> None:
    args = gui_main._build_parser().parse_args(["--render-theme-previews", "artifacts/gui"])
    assert args.theme == "warm"
    assert args.render_theme_previews == Path("artifacts/gui")


def test_gui_main_skips_runtime_cleanup_for_preview_render(monkeypatch, tmp_path: Path) -> None:
    seen: list[str] = []

    monkeypatch.setattr(gui_main, "ensure_standard_streams", lambda: None)
    monkeypatch.setattr(gui_main, "install_faulthandler_trace", lambda: None)
    monkeypatch.setattr(gui_main, "append_runtime_diagnostic_event", lambda *args, **kwargs: seen.append(args[0]))
    monkeypatch.setattr(gui_main.multiprocessing, "freeze_support", lambda: None)
    monkeypatch.setattr(
        gui_main.llama_runtime,
        "cleanup_stale_managed_llama_server_processes",
        lambda: seen.append("cleanup") or False,
    )
    monkeypatch.setattr(
        gui_main,
        "render_theme_previews",
        lambda output_dir: seen.append(str(output_dir)) or tuple(),
    )

    assert gui_main.main(["--render-theme-previews", str(tmp_path)]) == 0
    assert seen == [
        "gui_main_start",
        "gui_preview_render_start",
        str(tmp_path),
        "gui_preview_render_complete",
    ]
