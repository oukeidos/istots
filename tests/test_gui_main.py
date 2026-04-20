from pathlib import Path

from istots.gui_main import _build_parser


def test_gui_main_accepts_theme_selection() -> None:
    args = _build_parser().parse_args(["--theme", "poster"])
    assert args.theme == "poster"
    assert args.render_theme_previews is None


def test_gui_main_accepts_preview_render_directory() -> None:
    args = _build_parser().parse_args(["--render-theme-previews", "artifacts/gui"])
    assert args.theme == "warm"
    assert args.render_theme_previews == Path("artifacts/gui")
