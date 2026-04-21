from __future__ import annotations

import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from istots import __version__
from istots.app.convert import ConvertProgressSnapshot
from istots.app.setup import SetupProgressEvent, SetupRequest, SetupResult
from istots.gui.core import GuiRuntimeStatus
from istots.gui.qt_app import (
    TastingWindow,
    _apply_application_metadata,
    _status_shape_name,
    _wrap_message_box_text,
    list_gui_theme_ids,
    resolve_gui_theme,
)
from istots.runtime_prerequisites import RuntimePrerequisiteStatus


@pytest.fixture(autouse=True)
def _reset_llama_runtime_shutdown_flag() -> None:
    from istots import llama_runtime

    llama_runtime.clear_llama_server_process_shutdown_request()
    yield
    llama_runtime.clear_llama_server_process_shutdown_request()


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


def test_tasting_window_uses_packaged_gui_icon() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        assert not app.windowIcon().isNull()
        assert not window.windowIcon().isNull()
    finally:
        window.close()


def test_tasting_window_keeps_run_and_setup_as_separate_actions() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        assert window.primary_button.text() == "Run"
        assert window.setup_button.text() == "Set Up"
        assert window.primary_button.parent() is window.action_card
        assert window.setup_button.parent() is window.setup_lane
        assert window.refresh_button.parent() is window.test_lane
        assert window.setup_label.text() == "Setup"
        assert window.test_label.text() == "Test"
        assert window.runtime_variant_combo.currentText() == "Auto"
        assert window.runtime_variant_combo.findData("arm64/cpu") >= 0
    finally:
        window.close()


def test_tasting_window_relies_on_system_font_family_by_default() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        assert "font-family" not in window.styleSheet()
    finally:
        window.close()


def test_apply_application_metadata_sets_application_version() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    _apply_application_metadata(app)

    assert app.applicationName() == "istots"
    assert app.applicationVersion() == __version__


def test_tasting_window_title_includes_version() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)

    try:
        assert window.windowTitle() == f"istots {__version__}"
    finally:
        window.close()


def test_tasting_window_setup_progress_time_reports_elapsed_and_quiet_period(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    clock = {"now": 100.0}
    monkeypatch.setattr("istots.gui.qt_app.time.monotonic", lambda: clock["now"])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        window._active_task_title = "Setup"
        window._begin_setup_progress()
        assert window.progress_time.text() == "Still working... 00:00 elapsed"
        clock["now"] = 112.0
        window._refresh_setup_progress_display()
        assert window.progress_time.text() == "Still working... 00:12 elapsed"
    finally:
        window.close()


def test_tasting_window_setup_progress_event_keeps_time_visible(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    clock = {"now": 200.0}
    monkeypatch.setattr("istots.gui.qt_app.time.monotonic", lambda: clock["now"])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        window._active_task_title = "Setup"
        window._on_setup_progress_event(
            SetupProgressEvent(
                phase="model_setup",
                headline="Setup Assets",
                detail="Preparing model assets; downloads can stay quiet for a while",
                fraction=0.8,
            )
        )
        assert window.progress_detail.text().startswith("Setup Assets")
        assert window.progress_time.isVisible()
        assert window.progress_time.text() == "Still working... 00:00 elapsed"
    finally:
        window.close()


def test_tasting_window_close_event_allows_keep_working_during_active_task(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        window._thread = object()
        window._active_task_title = "Setup"
        monkeypatch.setattr(
            window,
            "_show_message_box",
            lambda **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
        )

        event = QtGui.QCloseEvent()
        window.closeEvent(event)

        assert event.isAccepted() is False
    finally:
        window._thread = None
        window.close()


def test_tasting_window_close_event_can_close_anyway_during_active_task(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        window._thread = object()
        window._active_task_title = "Run"
        closed: list[str] = []
        monkeypatch.setattr(
            window,
            "_show_message_box",
            lambda **kwargs: QtWidgets.QMessageBox.StandardButton.Close,
        )
        monkeypatch.setattr(
            window,
            "_force_close_active_task_for_exit",
            lambda: closed.append("forced"),
        )

        event = QtGui.QCloseEvent()
        window.closeEvent(event)

        assert event.isAccepted() is False
        assert closed == ["forced"]
        assert window.isVisible() is False
    finally:
        window._thread = None
        window.close()


def test_tasting_window_force_close_active_task_cleans_runtime_and_schedules_termination(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    class _FakeThread:
        def __init__(self) -> None:
            self.interruption_requested = False
            self.quit_called = False
            self.terminate_called = False
            self.wait_calls: list[int] = []

        def requestInterruption(self) -> None:
            self.interruption_requested = True

        def quit(self) -> None:
            self.quit_called = True

        def wait(self, timeout: int) -> bool:
            self.wait_calls.append(timeout)
            return False if len(self.wait_calls) == 1 else True

        def terminate(self) -> None:
            self.terminate_called = True

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        thread = _FakeThread()
        window._thread = thread
        window._active_task_cancel_event = threading.Event()
        cleaned: list[str] = []
        shutdown_requested: list[str] = []
        single_shots: list[tuple[int, object]] = []
        monkeypatch.setattr(
            "istots.llama_runtime.request_llama_server_process_shutdown",
            lambda: shutdown_requested.append("shutdown"),
        )
        monkeypatch.setattr(
            "istots.llama_runtime.cleanup_managed_llama_server_for_current_process",
            lambda: cleaned.append("runtime") or True,
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.QtCore.QTimer.singleShot",
            lambda delay_ms, callback: single_shots.append((delay_ms, callback)),
        )

        window._force_close_active_task_for_exit()

        assert window._closing_anyway is True
        assert window._active_task_cancel_event is not None
        assert window._active_task_cancel_event.is_set() is True
        assert shutdown_requested == ["shutdown"]
        assert cleaned == ["runtime"]
        assert thread.interruption_requested is True
        assert thread.quit_called is True
        assert thread.wait_calls == [250]
        assert thread.terminate_called is False
        assert len(single_shots) == 1
        assert single_shots[0][0] == 1500
    finally:
        window._thread = None
        window.close()


def test_tasting_window_force_close_termination_fallback_terminates_thread() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    class _FakeThread:
        def __init__(self) -> None:
            self.terminate_called = False
            self.wait_calls: list[int] = []

        def isFinished(self) -> bool:
            return False

        def terminate(self) -> None:
            self.terminate_called = True

        def wait(self, timeout: int) -> bool:
            self.wait_calls.append(timeout)
            return True

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        thread = _FakeThread()
        window._thread = thread
        window._closing_anyway = True

        window._terminate_active_task_thread_if_still_running()

        assert thread.terminate_called is True
        assert thread.wait_calls == [1000]
    finally:
        window._thread = None
        window.close()


def test_tasting_window_close_event_cleans_runtime_without_active_task(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        cleaned: list[str] = []
        monkeypatch.setattr(
            window,
            "_cleanup_runtime_for_exit",
            lambda: cleaned.append("runtime"),
        )

        event = QtGui.QCloseEvent()
        window.closeEvent(event)

        assert event.isAccepted() is True
        assert cleaned == ["runtime"]
    finally:
        window.close()


def test_tasting_window_setup_task_raises_when_final_runtime_check_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        request = SetupRequest(models_dir=tmp_path / "models", bootstrap_managed_runtime=True)
        emitted: list[SetupProgressEvent] = []

        monkeypatch.setattr(
            "istots.gui.qt_app.build_setup_request_for_variant",
            lambda **kwargs: request,
        )

        def _fake_execute_setup_request(_request, *, progress_callback, emit_completion_event, cancel_event):
            assert emit_completion_event is False
            assert cancel_event is None or hasattr(cancel_event, "is_set")
            progress_callback(
                SetupProgressEvent(
                    phase="model_setup",
                    headline="Setup Assets",
                    detail="Provisioning model assets",
                    fraction=0.8,
                )
            )
            return SetupResult(
                artifacts=SimpleNamespace(),
                custom_hf_bundle=False,
                custom_gguf_bundle=False,
                custom_qwen_bundle=False,
            )

        monkeypatch.setattr(
            "istots.gui.qt_app.execute_setup_request",
            _fake_execute_setup_request,
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.run_gui_doctor_check",
            lambda **kwargs: GuiRuntimeStatus(
                ready=False,
                headline="Check",
                detail="runtime test failed",
                missing_items=("ocr:startup_failed",),
            ),
        )

        with pytest.raises(RuntimeError, match="runtime test failed"):
            window._run_setup_task(emitted.append)

        assert [event.phase for event in emitted] == ["model_setup", "runtime_check"]
    finally:
        window.close()


def test_tasting_window_setup_task_uses_selected_runtime_variant(
    monkeypatch,
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        seen_variants: list[str] = []
        request = SetupRequest(models_dir=tmp_path / "models", bootstrap_managed_runtime=True)
        index = window.runtime_variant_combo.findData("x64/cpu")
        assert index >= 0
        window.runtime_variant_combo.setCurrentIndex(index)
        app.processEvents()

        monkeypatch.setattr(
            "istots.gui.qt_app.build_setup_request_for_variant",
            lambda **kwargs: seen_variants.append(kwargs["runtime_variant"]) or request,
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.execute_setup_request",
            lambda _request, *, progress_callback, emit_completion_event, cancel_event: SetupResult(
                artifacts=SimpleNamespace(),
                custom_hf_bundle=False,
                custom_gguf_bundle=False,
                custom_qwen_bundle=False,
            ),
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.run_gui_doctor_check",
            lambda **kwargs: GuiRuntimeStatus(
                ready=True,
                headline="Ready",
                detail="Runtime test passed.",
                missing_items=(),
            ),
        )

        window._run_setup_task(lambda event: None)

        assert seen_variants == ["x64/cpu"]
    finally:
        window.close()


def test_tasting_window_handle_setup_action_prompts_for_missing_prerequisite_and_cancels(
    monkeypatch,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        started: list[object] = []
        monkeypatch.setattr(
            "istots.gui.qt_app.missing_managed_runtime_prerequisites",
            lambda: (
                RuntimePrerequisiteStatus(
                    key="windows-msvc-v14-x64",
                    label="Microsoft Visual C++ Redistributable (x64)",
                    ok=False,
                    detail="missing",
                    remediation="install it",
                    installer_url="https://aka.ms/vs/17/release/vc_redist.x64.exe",
                ),
            ),
        )
        monkeypatch.setattr(
            window,
            "_show_message_box",
            lambda **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        monkeypatch.setattr(window, "_start_task", lambda **kwargs: started.append(kwargs))

        window._handle_setup_action()

        assert started == []
    finally:
        window.close()


def test_tasting_window_handle_setup_action_confirms_prerequisite_install(
    monkeypatch,
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        monkeypatch.setattr(
            "istots.gui.qt_app.missing_managed_runtime_prerequisites",
            lambda: (
                RuntimePrerequisiteStatus(
                    key="windows-msvc-v14-x64",
                    label="Microsoft Visual C++ Redistributable (x64)",
                    ok=False,
                    detail="missing",
                    remediation="install it",
                    installer_url="https://aka.ms/vs/17/release/vc_redist.x64.exe",
                ),
            ),
        )
        monkeypatch.setattr(
            window,
            "_show_message_box",
            lambda **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )

        started: list[dict[str, object]] = []
        monkeypatch.setattr(window, "_start_task", lambda **kwargs: started.append(kwargs))

        window._handle_setup_action()

        assert len(started) == 1
        setup_kwargs = started[0]
        assert setup_kwargs["title"] == "Setup"
        assert setup_kwargs["cancel_event"] is not None

        seen_install_flags: list[bool] = []
        request = SetupRequest(models_dir=tmp_path / "models", bootstrap_managed_runtime=True)
        monkeypatch.setattr(
            "istots.gui.qt_app.build_setup_request_for_variant",
            lambda **kwargs: seen_install_flags.append(kwargs["install_prerequisites"]) or request,
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.execute_setup_request",
            lambda _request, *, progress_callback, emit_completion_event, cancel_event: SetupResult(
                artifacts=SimpleNamespace(),
                custom_hf_bundle=False,
                custom_gguf_bundle=False,
                custom_qwen_bundle=False,
            ),
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.run_gui_doctor_check",
            lambda **kwargs: GuiRuntimeStatus(
                ready=True,
                headline="Ready",
                detail="Runtime test passed.",
                missing_items=(),
            ),
        )

        setup_kwargs["fn"](lambda event: None)

        assert seen_install_flags == [True]
    finally:
        window.close()


def test_tasting_window_start_runtime_check_uses_cancel_event(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        started: list[dict[str, object]] = []
        monkeypatch.setattr(window, "_start_task", lambda **kwargs: started.append(kwargs))

        window._start_runtime_check()

        assert len(started) == 1
        assert started[0]["title"] == "Check"
        assert started[0]["cancel_event"] is not None
    finally:
        window.close()


def test_tasting_window_runtime_facts_show_target_and_active_runtime() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        index = window.runtime_variant_combo.findData("x64/vulkan")
        assert index >= 0
        window.runtime_variant_combo.setCurrentIndex(index)
        window._apply_runtime_facts(
            GuiRuntimeStatus(
                ready=True,
                headline="Ready",
                detail="",
                missing_items=(),
                runtime_binary_path=Path("C:/runtime/llama-server.exe"),
                runtime_source="managed",
                runtime_release_tag="b8860",
                runtime_variant_id="x64/vulkan",
            )
        )
        app.processEvents()

        assert "target x64/vulkan" in window.setup_summary.full_text()
        assert "Managed b8860 [x64/vulkan]" in window.setup_summary.full_text()
        assert "\n" not in window.setup_summary.full_text()
        assert "Path:" in window.setup_summary.toolTip()
    finally:
        window.close()


def test_tasting_window_setup_finish_refreshes_runtime_facts_after_variant_change(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        index = window.runtime_variant_combo.findData("x64/cpu")
        assert index >= 0
        window.runtime_variant_combo.setCurrentIndex(index)
        app.processEvents()

        monkeypatch.setattr(
            "istots.gui.qt_app.probe_runtime_status",
            lambda: GuiRuntimeStatus(
                ready=True,
                headline="Ready",
                detail="",
                missing_items=(),
                runtime_binary_path=Path("C:/runtime/llama-server.exe"),
                runtime_source="managed",
                runtime_release_tag="b9999",
                runtime_variant_id="x64/cpu",
            ),
        )

        window._on_setup_finished(
            SetupResult(
                artifacts=SimpleNamespace(),
                custom_hf_bundle=False,
                custom_gguf_bundle=False,
                custom_qwen_bundle=False,
            )
        )

        assert "target x64/cpu" in window.setup_summary.full_text()
        assert "Managed b9999 [x64/cpu]" in window.setup_summary.full_text()
    finally:
        window.close()


def test_tasting_window_check_failure_routes_to_popup(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        seen: list[tuple[str, str]] = []
        monkeypatch.setattr(
            window,
            "_show_message_box",
            lambda **kwargs: seen.append((kwargs["title"], kwargs["message"])),
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.probe_runtime_status",
            lambda: GuiRuntimeStatus(
                ready=False,
                headline="Setup",
                detail="runtime still unavailable",
                missing_items=("llama-server",),
            ),
        )

        window._on_runtime_check_finished(
            GuiRuntimeStatus(
                ready=False,
                headline="Check",
                detail="runtime test failed",
                missing_items=("ocr:startup_failed",),
            )
        )

        assert seen == [("Test", "runtime test failed")]
    finally:
        window.close()


def test_tasting_window_message_box_returns_standard_result(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        result = window._show_message_box(
            icon=QtWidgets.QMessageBox.Icon.Warning,
            title="Test",
            message="popup path check",
        )
        assert result == QtWidgets.QMessageBox.StandardButton.Ok
    finally:
        window.close()


def test_tasting_window_message_box_reflows_text_into_the_first_column(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        captured: dict[str, object] = {}

        def _fake_exec(box: QtWidgets.QMessageBox) -> int:
            box.show()
            app.processEvents()
            label = box.findChild(QtWidgets.QLabel, "qt_msgbox_label")
            informative = box.findChild(QtWidgets.QLabel, "qt_msgbox_informativelabel")
            assert label is not None
            assert informative is not None
            captured["x"] = label.geometry().x()
            captured["primary"] = label.text()
            captured["informative"] = informative.text()
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _fake_exec)

        result = window._show_message_box(
            icon=QtWidgets.QMessageBox.Icon.Warning,
            title="Setup",
            message=(
                "managed llama.cpp runtime failed startup validation.\n"
                "Binary: C:\\Users\\svmagic\\AppData\\Local\\istots\\managed\\runtime\\llama.cpp\\b8860\\x64-vulkan\\llama-server.exe\n"
                "Probe --version failed with exit=3221225477.\n"
                "Probe --help failed with exit=3221225477."
            ),
        )

        assert result == QtWidgets.QMessageBox.StandardButton.Ok
        assert captured["x"] == 11
        assert str(captured["primary"]).replace("\n", " ") == (
            "managed llama.cpp runtime failed startup validation."
        )
        assert "Binary:" in str(captured["informative"])
    finally:
        window.close()


def test_tasting_window_message_box_sizes_long_buttons_and_styles_menus(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        captured: dict[str, int] = {}

        def _fake_exec(box: QtWidgets.QMessageBox) -> int:
            box.show()
            app.processEvents()
            metrics = QtGui.QFontMetrics(box.font())
            for button in box.findChildren(QtWidgets.QPushButton):
                text = button.text().replace("&", "")
                if "Details" in text or text in {"Stay Open", "Close Now"}:
                    captured[text] = button.minimumWidth() - metrics.horizontalAdvance(text)
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _fake_exec)

        result = window._show_message_box(
            icon=QtWidgets.QMessageBox.Icon.Warning,
            title="Close App?",
            message=(
                "managed llama.cpp runtime failed startup validation.\n"
                "Binary: C:\\Users\\svmagic\\AppData\\Local\\istots\\managed\\runtime\\llama.cpp\\b8860\\x64-vulkan\\llama-server.exe\n"
                "Release: b8860\n"
                "Variant: x64/vulkan\n"
                "Probe --version failed with exit=3221225477.\n"
                "Probe --help failed with exit=3221225477.\n"
                "A temporary loader or security scan race is possible.\n"
                "Retry setup for the same target, or choose a different runtime target.\n"
                "If this persists, copy the details and inspect the managed runtime folder."
            ),
            buttons=(
                QtWidgets.QMessageBox.StandardButton.Close
                | QtWidgets.QMessageBox.StandardButton.Cancel
            ),
            default_button=QtWidgets.QMessageBox.StandardButton.Cancel,
            button_text_overrides={
                QtWidgets.QMessageBox.StandardButton.Cancel: "Stay Open",
                QtWidgets.QMessageBox.StandardButton.Close: "Close Now",
            },
        )

        assert result == QtWidgets.QMessageBox.StandardButton.Ok
        assert "QMenu {" in window.styleSheet()
        assert captured["Stay Open"] >= 40
        assert captured["Close Now"] >= 40
        details_key = next(key for key in captured if "Details" in key)
        assert captured[details_key] >= 40
    finally:
        window.close()


def test_tasting_window_message_box_wraps_detailed_text_editor(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        captured: dict[str, object] = {}

        def _fake_exec(box: QtWidgets.QMessageBox) -> int:
            box.show()
            app.processEvents()
            details_button = next(
                button
                for button in box.findChildren(QtWidgets.QPushButton)
                if "Details" in button.text().replace("&", "")
            )
            details_button.click()
            app.processEvents()
            details_edit = box.findChild(QtWidgets.QTextEdit)
            assert details_edit is not None
            captured["wrap_mode"] = details_edit.lineWrapMode()
            captured["h_scroll_policy"] = details_edit.horizontalScrollBarPolicy()
            captured["minimum_width"] = details_edit.minimumWidth()
            captured["minimum_height"] = details_edit.minimumHeight()
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _fake_exec)

        result = window._show_message_box(
            icon=QtWidgets.QMessageBox.Icon.Warning,
            title="Setup",
            message=(
                "managed llama.cpp runtime failed startup validation.\n"
                "Binary: C:\\Users\\svmagic\\AppData\\Local\\istots\\managed\\runtime\\llama.cpp\\b8860\\x64-vulkan\\llama-server.exe\n"
                "Release: b8860\n"
                "Variant: x64/vulkan\n"
                "Probe --version failed with exit=3221225477.\n"
                "Probe --help failed with exit=3221225477.\n"
                "A temporary loader or security scan race is possible.\n"
                "Retry setup for the same target, or choose a different runtime target.\n"
                "If this persists, copy the details and inspect the managed runtime folder."
            ),
        )

        assert result == QtWidgets.QMessageBox.StandardButton.Ok
        assert captured["wrap_mode"] == QtWidgets.QTextEdit.LineWrapMode.WidgetWidth
        assert captured["h_scroll_policy"] == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert captured["minimum_width"] >= 640
        assert captured["minimum_height"] >= 180
    finally:
        window.close()


def test_wrap_message_box_text_inserts_breaks_for_long_content() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    text = (
        "managed llama.cpp runtime failed startup validation. "
        "Binary: C:\\Users\\svmagic\\AppData\\Local\\istots\\managed\\runtime\\llama.cpp\\b8858\\x64-vulkan\\llama-server.exe "
        "Probe --version failed with exit=3221225477."
    )

    wrapped = _wrap_message_box_text(text, font=app.font(), max_width_px=320)

    assert "\n" in wrapped


def test_tasting_window_setup_failure_updates_main_window_feedback(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TastingWindow(theme_id="warm", preview_fixture=True)
    window.show()
    app.processEvents()

    try:
        monkeypatch.setattr(
            window,
            "_show_message_box",
            lambda **kwargs: QtWidgets.QMessageBox.StandardButton.Ok,
        )
        monkeypatch.setattr(
            "istots.gui.qt_app.probe_runtime_status",
            lambda: GuiRuntimeStatus(
                ready=True,
                headline="Ready",
                detail="runtime assets exist",
                missing_items=(),
            ),
        )

        window._clear_run_feedback()
        window._active_task_title = "Setup"
        window._on_task_failed("Setup", "runtime validation failed")
        window._on_task_finished()

        assert window.test_summary.full_text().startswith("Failed")
        assert window.test_summary.toolTip() == "runtime validation failed"
        assert window.setup_summary.full_text().startswith("Needed")
        assert "runtime validation failed" in window.setup_summary.toolTip()
        assert not window.progress.isVisible()
        assert not window.progress_detail.isVisible()
        assert window.progress_detail.text() == ""
    finally:
        window.close()
