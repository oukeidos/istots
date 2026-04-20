from __future__ import annotations

import os

from istots.app.convert import ConvertProgressSnapshot
from istots.gui.qt_app import TastingWindow, _status_shape_name, list_gui_theme_ids, resolve_gui_theme


def test_status_glyph_shapes_distinguish_states_beyond_color() -> None:
    assert _status_shape_name("setup") == "triangle"
    assert _status_shape_name("ready") == "circle-check"
    assert _status_shape_name("idle") == "ring"
    assert _status_shape_name("busy") == "diamond"
    assert _status_shape_name("ok") == "circle-check"
    assert _status_shape_name("fail") == "square-x"


def test_gui_themes_expose_all_three_preview_variants() -> None:
    assert list_gui_theme_ids() == ("warm", "warm-glass", "poster")
    assert resolve_gui_theme("warm").label.startswith("1 ")
    assert resolve_gui_theme("warm-glass").label.startswith("1+2 ")
    assert resolve_gui_theme("poster").label.startswith("3 ")
    assert resolve_gui_theme("missing").key == "warm"


def test_warm_theme_is_the_flatter_small_screen_friendly_base() -> None:
    warm = resolve_gui_theme("warm")
    assert warm.shadow_blur == 0
    assert warm.shadow_offset_y == 0
    assert warm.outer_margin <= 22
    assert warm.icon_button_size <= 46
    assert warm.base_font_size == 16
    assert warm.title_font_size == 16
    assert warm.progress_font_size == 16
    assert warm.time_font_size == 16


def test_warm_window_keeps_a_fixed_default_height_when_run_feedback_appears() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        window._clear_run_feedback()
        window._apply_window_fit()
        app.processEvents()
        idle_height = window.height()
        idle_button_pos = window.primary_button.mapTo(window.centralWidget(), QtCore.QPoint(0, 0))
        idle_action_y = window.action_card.geometry().y()
        idle_action_height = window.action_card.geometry().height()
        bottom_margin = window._outer_layout.contentsMargins().bottom()
        theme = resolve_gui_theme("warm")

        window._set_run_feedback(
            state="running",
            detail="OCR 30/60 62%",
            time_text="00:24 / est. 00:15 left",
            value=620,
        )
        app.processEvents()

        assert idle_height >= 610
        assert window.height() == idle_height
        run_button_pos = window.primary_button.mapTo(window.centralWidget(), QtCore.QPoint(0, 0))
        assert run_button_pos.y() == idle_button_pos.y()
        assert window.action_card.geometry().y() == idle_action_y
        assert window.action_card.geometry().height() == idle_action_height
        assert (
            window.centralWidget().height()
            - (window.action_card.geometry().y() + window.action_card.geometry().height())
            - bottom_margin
            == 0
        )
        assert window.input_edit.height() > theme.icon_button_size
        assert window.output_edit.height() > theme.icon_button_size
        assert window.primary_button.minimumHeight() >= 56
    finally:
        window.close()


def test_progress_time_marks_remaining_as_estimated() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        formatted = window._format_progress_time(
            ConvertProgressSnapshot(
                phase="ocr_progress",
                headline="OCR",
                detail="30/60",
                fraction=0.62,
                elapsed_sec=24.0,
                eta_sec=15.0,
            )
        )
        assert formatted == "00:24 / est. 00:15 left"
    finally:
        window.close()
