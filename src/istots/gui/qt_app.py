from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time
from typing import Literal

from istots import __version__
from istots.app.convert import (
    ConvertArgumentError,
    ConvertProgressEstimator,
    ConvertProgressEvent,
    ConvertProgressSnapshot,
    ConvertPreparationError,
    ConvertResult,
    execute_convert_plan,
    plan_convert_request,
)
from istots.app.setup import SetupProgressEvent, SetupResult, execute_setup_request
from istots.gui.bootstrap_windows import MANUAL_RUNTIME_VARIANTS, record_managed_runtime_validation
from istots.gui.core import (
    GuiPrimaryAction,
    GuiRuntimeStatus,
    GuiScreenState,
    build_fast_convert_request,
    build_setup_request_for_variant,
    derive_primary_action,
    derive_setup_action,
    format_check_summary,
    format_runtime_facts,
    format_setup_summary,
    probe_runtime_status,
    run_gui_doctor_check,
    suggest_output_srt_path,
)
from istots.runtime_diagnostics import append_runtime_diagnostic_event
from istots.runtime_prerequisites import (
    format_missing_managed_runtime_prerequisites,
    missing_managed_runtime_prerequisites,
)
from istots.resources import iter_gui_icon_png_payloads

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover - exercised via runtime guard
    QtCore = None
    QtGui = None
    QtWidgets = None
    _QT_IMPORT_ERROR = exc
else:
    _QT_IMPORT_ERROR = None


class MissingGuiDependencyError(RuntimeError):
    pass


GuiThemeId = Literal["warm", "warm-glass", "poster"]


@dataclass(frozen=True)
class GuiThemeSpec:
    key: GuiThemeId
    label: str
    app_background: str
    card_background: str
    status_card_background: str
    action_card_background: str
    card_border: str
    text_color: str
    heading_color: str
    muted_text: str
    divider: str
    input_background: str
    input_border: str
    input_focus: str
    secondary_button_background: str
    secondary_button_hover: str
    secondary_button_text: str
    primary_background: str
    primary_hover: str
    primary_disabled: str
    primary_disabled_text: str
    progress_background: str
    progress_chunk: str
    checkbox_text: str
    selection_background: str
    base_font_size: int
    title_font_size: int
    primary_font_size: int
    progress_font_size: int
    time_font_size: int
    card_radius: int
    button_radius: int
    input_radius: int
    icon_button_size: int
    progress_height: int
    outer_margin: int
    outer_spacing: int
    card_spacing: int
    status_padding: tuple[int, int, int, int]
    fields_padding: tuple[int, int, int, int]
    action_padding: tuple[int, int, int, int]
    field_spacing: int
    action_spacing: int
    shadow_color: str
    shadow_blur: int
    shadow_offset_y: int


_GUI_THEMES: dict[GuiThemeId, GuiThemeSpec] = {
    "warm": GuiThemeSpec(
        key="warm",
        label="1 Warm Layered Utility",
        app_background="qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f7f1ea, stop:0.62 #efe6da, stop:1 #e8dbcd)",
        card_background="#fffaf4",
        status_card_background="#fffaf4",
        action_card_background="#fff7f0",
        card_border="#e6d7c4",
        text_color="#201a16",
        heading_color="#231c18",
        muted_text="#6d5b50",
        divider="#e7d9c8",
        input_background="#fffdf9",
        input_border="#dccbb7",
        input_focus="#d6782b",
        secondary_button_background="#ecdfcf",
        secondary_button_hover="#e7d7c1",
        secondary_button_text="#231c18",
        primary_background="#d6782b",
        primary_hover="#c36a22",
        primary_disabled="#e6c3a3",
        primary_disabled_text="#fff3ea",
        progress_background="#efe4d8",
        progress_chunk="#d6782b",
        checkbox_text="#2f2621",
        selection_background="#d6782b",
        base_font_size=16,
        title_font_size=16,
        primary_font_size=18,
        progress_font_size=16,
        time_font_size=16,
        card_radius=20,
        button_radius=15,
        input_radius=13,
        icon_button_size=46,
        progress_height=8,
        outer_margin=22,
        outer_spacing=16,
        card_spacing=12,
        status_padding=(20, 18, 20, 18),
        fields_padding=(20, 20, 20, 20),
        action_padding=(20, 18, 20, 18),
        field_spacing=14,
        action_spacing=12,
        shadow_color="transparent",
        shadow_blur=0,
        shadow_offset_y=0,
    ),
    "warm-glass": GuiThemeSpec(
        key="warm-glass",
        label="1+2 Warm Layered + Quiet Glass",
        app_background="qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f3ece6, stop:0.46 #e7edf0, stop:1 #d7e1e7)",
        card_background="rgba(255, 251, 247, 204)",
        status_card_background="rgba(255, 252, 248, 199)",
        action_card_background="rgba(255, 248, 243, 219)",
        card_border="rgba(255, 255, 255, 189)",
        text_color="#1c2328",
        heading_color="#162026",
        muted_text="#67737b",
        divider="rgba(255, 255, 255, 158)",
        input_background="rgba(255, 255, 255, 199)",
        input_border="rgba(192, 182, 174, 184)",
        input_focus="#cc7147",
        secondary_button_background="rgba(255, 255, 255, 143)",
        secondary_button_hover="rgba(255, 255, 255, 189)",
        secondary_button_text="#223037",
        primary_background="#ca7148",
        primary_hover="#b9603c",
        primary_disabled="#e7c4b4",
        primary_disabled_text="#fff7f4",
        progress_background="rgba(255, 255, 255, 133)",
        progress_chunk="qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d28257, stop:1 #bf6045)",
        checkbox_text="#243038",
        selection_background="#ca7148",
        base_font_size=15,
        title_font_size=18,
        primary_font_size=20,
        progress_font_size=14,
        time_font_size=13,
        card_radius=28,
        button_radius=18,
        input_radius=16,
        icon_button_size=48,
        progress_height=10,
        outer_margin=32,
        outer_spacing=20,
        card_spacing=15,
        status_padding=(27, 23, 27, 23),
        fields_padding=(28, 28, 28, 28),
        action_padding=(28, 25, 28, 25),
        field_spacing=16,
        action_spacing=15,
        shadow_color="rgba(69, 99, 121, 56)",
        shadow_blur=48,
        shadow_offset_y=16,
    ),
    "poster": GuiThemeSpec(
        key="poster",
        label="3 Typography-First Poster Tool",
        app_background="qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fbf4ea, stop:0.65 #f1e1d0, stop:1 #e6d0bf)",
        card_background="#fff9f1",
        status_card_background="#fffdf7",
        action_card_background="#fbefe4",
        card_border="#d8c0ab",
        text_color="#16110f",
        heading_color="#14100f",
        muted_text="#615146",
        divider="#dec8b5",
        input_background="#fffaf3",
        input_border="#cfb79e",
        input_focus="#b14a2c",
        secondary_button_background="#efe0d0",
        secondary_button_hover="#e5d0bc",
        secondary_button_text="#181210",
        primary_background="#a64027",
        primary_hover="#8f361f",
        primary_disabled="#dfb7ab",
        primary_disabled_text="#fff7f2",
        progress_background="#eadccf",
        progress_chunk="#a64027",
        checkbox_text="#241c19",
        selection_background="#b14a2c",
        base_font_size=16,
        title_font_size=22,
        primary_font_size=22,
        progress_font_size=15,
        time_font_size=14,
        card_radius=30,
        button_radius=20,
        input_radius=18,
        icon_button_size=52,
        progress_height=12,
        outer_margin=36,
        outer_spacing=24,
        card_spacing=18,
        status_padding=(30, 25, 30, 25),
        fields_padding=(30, 30, 30, 30),
        action_padding=(30, 26, 30, 26),
        field_spacing=18,
        action_spacing=16,
        shadow_color="rgba(88, 45, 18, 41)",
        shadow_blur=32,
        shadow_offset_y=10,
    ),
}


def list_gui_theme_ids() -> tuple[GuiThemeId, ...]:
    return tuple(_GUI_THEMES)


def resolve_gui_theme(theme_id: str | None) -> GuiThemeSpec:
    if theme_id in _GUI_THEMES:
        return _GUI_THEMES[theme_id]
    return _GUI_THEMES["warm"]


@dataclass(frozen=True)
class _TaskFailure:
    title: str
    message: str


@dataclass(frozen=True)
class _RunFeedback:
    state: str = "idle"
    detail: str = ""
    time_text: str = ""
    value: int = 0
    visible: bool = False


@dataclass(frozen=True)
class _CheckFeedback:
    state: str = "idle"
    summary: str = "Not tested"
    detail: str = ""


def _ensure_qt() -> None:
    if _QT_IMPORT_ERROR is None:
        return
    raise MissingGuiDependencyError(
        "GUI runtime requires PySide6. Install it with `uv sync --extra gui` first."
    ) from _QT_IMPORT_ERROR


def _status_shape_name(state: str) -> str:
    return {
        "ready": "circle-check",
        "setup": "triangle",
        "idle": "ring",
        "busy": "diamond",
        "ok": "circle-check",
        "fail": "square-x",
    }.get(state, "ring")


def _default_check_summary(state: str) -> str:
    return {
        "idle": "Not tested",
        "busy": "Testing",
        "ok": "Passed",
        "fail": "Failed",
    }.get(state, "Not tested")


def _message_box_standard_pixmap(style, icon, size):
    if QtWidgets is None:
        return None
    icon_map = {
        QtWidgets.QMessageBox.Icon.Information: QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation,
        QtWidgets.QMessageBox.Icon.Warning: QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        QtWidgets.QMessageBox.Icon.Critical: QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical,
        QtWidgets.QMessageBox.Icon.Question: QtWidgets.QStyle.StandardPixmap.SP_MessageBoxQuestion,
    }
    standard_pixmap = icon_map.get(icon)
    if standard_pixmap is None:
        return None
    return style.standardIcon(standard_pixmap).pixmap(size, size)


def _wrap_message_box_text(text: str, *, font, max_width_px: int) -> str:
    if QtGui is None or not text:
        return text

    metrics = QtGui.QFontMetrics(font)

    def _split_long_token(token: str) -> list[str]:
        if not token:
            return [token]
        chunks: list[str] = []
        current = ""
        for character in token:
            candidate = current + character
            if current and metrics.horizontalAdvance(candidate) > max_width_px:
                chunks.append(current)
                current = character
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    wrapped_lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph:
            wrapped_lines.append("")
            continue
        current_line = ""
        for token in paragraph.split(" "):
            token_parts = _split_long_token(token)
            for part in token_parts:
                candidate = part if not current_line else f"{current_line} {part}"
                if metrics.horizontalAdvance(candidate) <= max_width_px:
                    current_line = candidate
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = part
        if current_line:
            wrapped_lines.append(current_line)
    return "\n".join(wrapped_lines)


def _message_box_max_width(widget, *, detailed: bool = False) -> int:
    if QtWidgets is None:
        return 640 if detailed else 520
    screen = None
    if widget is not None:
        handle = widget.windowHandle()
        if handle is not None:
            screen = handle.screen()
        if screen is None:
            screen = widget.screen()
    if screen is None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
    available_width = screen.availableGeometry().width() if screen is not None else 920
    if detailed:
        return max(360, min(720, available_width - 120))
    return max(300, min(620, available_width - 140))


def _apply_message_box_button_widths(box) -> None:
    if QtWidgets is None or QtGui is None:
        return
    metrics = QtGui.QFontMetrics(box.font())
    for button in box.findChildren(QtWidgets.QPushButton):
        text = button.text().replace("&", "").strip()
        if not text:
            continue
        min_width = max(112, metrics.horizontalAdvance(text) + 52)
        button.setMinimumWidth(min_width)
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )


def _split_message_box_content(
    message: str,
    *,
    font,
    max_width_px: int,
) -> tuple[str, str, str]:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ("", "", "")
    primary_text = _wrap_message_box_text(lines[0], font=font, max_width_px=max_width_px)
    if len(lines) == 1:
        return (primary_text, "", "")

    informative_source = "\n".join(lines[1:])
    informative_wrapped = _wrap_message_box_text(
        informative_source,
        font=font,
        max_width_px=max_width_px,
    )
    if len(informative_wrapped.splitlines()) <= 5 and len(message) <= 320:
        return (primary_text, informative_wrapped, "")

    preview_source = "\n".join(lines[1:3])
    informative_preview = _wrap_message_box_text(
        preview_source,
        font=font,
        max_width_px=max_width_px,
    )
    return (
        primary_text,
        informative_preview,
        message,
    )


def _collapse_inline_status_text(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _format_setup_progress_label_text(
    event: SetupProgressEvent,
    *,
    max_chars: int = 120,
) -> tuple[str, str]:
    full_text = " ".join(
        part
        for part in (
            _collapse_inline_status_text(event.headline),
            _collapse_inline_status_text(event.detail),
        )
        if part
    )
    if not full_text:
        return ("", "")
    if len(full_text) <= max_chars:
        return (full_text, full_text)
    return (full_text[: max_chars - 3].rstrip() + "...", full_text)


def _build_gui_icon():
    if QtGui is None:
        return None

    icon = QtGui.QIcon()
    try:
        payloads = iter_gui_icon_png_payloads()
    except OSError:
        return icon

    for payload in payloads:
        pixmap = QtGui.QPixmap()
        if pixmap.loadFromData(payload, "PNG"):
            icon.addPixmap(pixmap)
    return icon


def _apply_application_metadata(app) -> None:
    app.setApplicationName("istots")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("istots")
    icon = _build_gui_icon()
    if icon is not None and not icon.isNull():
        app.setWindowIcon(icon)


if QtCore is not None:  # pragma: no branch
    class _FunctionWorker(QtCore.QObject):
        progressed = QtCore.Signal(object)
        succeeded = QtCore.Signal(object)
        failed = QtCore.Signal(str, str)
        finished = QtCore.Signal()

        def __init__(self, title: str, fn) -> None:
            super().__init__()
            self._title = title
            self._fn = fn

        @QtCore.Slot()
        def run(self) -> None:
            try:
                result = self._fn()
            except Exception as exc:
                self.failed.emit(self._title, str(exc))
            else:
                self.succeeded.emit(result)
            finally:
                self.finished.emit()


if QtWidgets is not None:  # pragma: no branch
    class _StatusGlyph(QtWidgets.QWidget):
        def __init__(self, state: str = "idle", parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self._state = state
            self.setFixedSize(16, 16)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.setAccessibleName(state)

        def sizeHint(self) -> QtCore.QSize:
            return QtCore.QSize(16, 16)

        def minimumSizeHint(self) -> QtCore.QSize:
            return self.sizeHint()

        def set_state(self, state: str) -> None:
            if self._state == state:
                return
            self._state = state
            self.setAccessibleName(state)
            self.update()

        def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            rect = QtCore.QRectF(1.5, 1.5, self.width() - 3.0, self.height() - 3.0)
            state = self._state
            shape = _status_shape_name(state)
            fill = self._fill_color(state)
            foreground = QtGui.QColor("#fffaf4")

            if shape == "ring":
                painter.setPen(QtGui.QPen(QtGui.QColor("#9d8f84"), 1.7))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawEllipse(rect.adjusted(1.0, 1.0, -1.0, -1.0))
                return

            if shape == "circle-check":
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawEllipse(rect)
                self._draw_check(painter, rect, foreground)
                return

            if shape == "triangle":
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawPolygon(
                    QtGui.QPolygonF(
                        [
                            QtCore.QPointF(rect.center().x(), rect.top()),
                            QtCore.QPointF(rect.right(), rect.bottom()),
                            QtCore.QPointF(rect.left(), rect.bottom()),
                        ]
                    )
                )
                self._draw_center_dot(painter, rect, foreground)
                return

            if shape == "diamond":
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawPolygon(
                    QtGui.QPolygonF(
                        [
                            QtCore.QPointF(rect.center().x(), rect.top()),
                            QtCore.QPointF(rect.right(), rect.center().y()),
                            QtCore.QPointF(rect.center().x(), rect.bottom()),
                            QtCore.QPointF(rect.left(), rect.center().y()),
                        ]
                    )
                )
                self._draw_center_dot(painter, rect, foreground)
                return

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 3.5, 3.5)
            self._draw_cross(painter, rect, foreground)

        def _fill_color(self, state: str) -> QtGui.QColor:
            return {
                "ready": QtGui.QColor("#1b6a42"),
                "setup": QtGui.QColor("#c78325"),
                "busy": QtGui.QColor("#4d62c7"),
                "ok": QtGui.QColor("#1b6a42"),
                "fail": QtGui.QColor("#b94a2f"),
            }.get(state, QtGui.QColor("#d8cbbb"))

        def _draw_check(
            self,
            painter: QtGui.QPainter,
            rect: QtCore.QRectF,
            color: QtGui.QColor,
        ) -> None:
            painter.setPen(
                QtGui.QPen(
                    color,
                    1.8,
                    QtCore.Qt.PenStyle.SolidLine,
                    QtCore.Qt.PenCapStyle.RoundCap,
                    QtCore.Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawLine(
                QtCore.QPointF(rect.left() + rect.width() * 0.25, rect.top() + rect.height() * 0.56),
                QtCore.QPointF(rect.left() + rect.width() * 0.44, rect.top() + rect.height() * 0.74),
            )
            painter.drawLine(
                QtCore.QPointF(rect.left() + rect.width() * 0.44, rect.top() + rect.height() * 0.74),
                QtCore.QPointF(rect.left() + rect.width() * 0.76, rect.top() + rect.height() * 0.32),
            )

        def _draw_center_dot(
            self,
            painter: QtGui.QPainter,
            rect: QtCore.QRectF,
            color: QtGui.QColor,
        ) -> None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            dot_rect = QtCore.QRectF(
                rect.center().x() - 1.9,
                rect.center().y() - 1.9,
                3.8,
                3.8,
            )
            painter.drawEllipse(dot_rect)

        def _draw_cross(
            self,
            painter: QtGui.QPainter,
            rect: QtCore.QRectF,
            color: QtGui.QColor,
        ) -> None:
            painter.setPen(
                QtGui.QPen(
                    color,
                    1.8,
                    QtCore.Qt.PenStyle.SolidLine,
                    QtCore.Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawLine(
                QtCore.QPointF(rect.left() + rect.width() * 0.28, rect.top() + rect.height() * 0.28),
                QtCore.QPointF(rect.left() + rect.width() * 0.72, rect.top() + rect.height() * 0.72),
            )
            painter.drawLine(
                QtCore.QPointF(rect.left() + rect.width() * 0.72, rect.top() + rect.height() * 0.28),
                QtCore.QPointF(rect.left() + rect.width() * 0.28, rect.top() + rect.height() * 0.72),
            )

    class _MaskCheckBox(QtWidgets.QCheckBox):
        def __init__(
            self,
            text: str,
            *,
            theme: GuiThemeSpec,
            parent: QtWidgets.QWidget | None = None,
        ) -> None:
            super().__init__(text, parent)
            self._theme = theme
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.setMinimumHeight(28)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )

        def sizeHint(self) -> QtCore.QSize:
            metrics = self.fontMetrics()
            width = 30 + metrics.horizontalAdvance(self.text()) + 8
            height = max(28, metrics.height() + 8)
            return QtCore.QSize(width, height)

        def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            rect = self.rect()
            indicator = QtCore.QRectF(0.5, (rect.height() - 18) / 2.0, 18, 18)

            if self.isEnabled():
                border = QtGui.QColor(self._theme.input_border)
                background = QtGui.QColor(self._theme.input_background)
                text_color = QtGui.QColor(self._theme.checkbox_text)
                accent = QtGui.QColor(self._theme.primary_background)
            else:
                border = QtGui.QColor(self._theme.divider)
                background = QtGui.QColor(self._theme.card_background)
                text_color = QtGui.QColor(self._theme.muted_text)
                accent = QtGui.QColor(self._theme.primary_disabled)

            painter.setPen(QtGui.QPen(border, 1.4))
            painter.setBrush(background)
            painter.drawRoundedRect(indicator, 5, 5)

            if self.isChecked():
                inner = indicator.adjusted(1.2, 1.2, -1.2, -1.2)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(accent)
                painter.drawRoundedRect(inner, 4, 4)
                painter.setPen(
                    QtGui.QPen(
                        QtGui.QColor("#fffaf4"),
                        1.8,
                        QtCore.Qt.PenStyle.SolidLine,
                        QtCore.Qt.PenCapStyle.RoundCap,
                        QtCore.Qt.PenJoinStyle.RoundJoin,
                    )
                )
                painter.drawLine(
                    QtCore.QPointF(inner.left() + inner.width() * 0.24, inner.top() + inner.height() * 0.56),
                    QtCore.QPointF(inner.left() + inner.width() * 0.43, inner.top() + inner.height() * 0.74),
                )
                painter.drawLine(
                    QtCore.QPointF(inner.left() + inner.width() * 0.43, inner.top() + inner.height() * 0.74),
                    QtCore.QPointF(inner.left() + inner.width() * 0.76, inner.top() + inner.height() * 0.32),
                )

            painter.setPen(text_color)
            text_rect = QtCore.QRectF(30, 0, rect.width() - 30, rect.height())
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                self.text(),
            )

    class _ElidedLabel(QtWidgets.QLabel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._full_text = ""
            self.setWordWrap(False)
            self.setMinimumWidth(0)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        def set_full_text(self, text: str, *, tooltip: str = "") -> None:
            self._full_text = text
            self.setToolTip(tooltip)
            self._apply_elision()

        def full_text(self) -> str:
            return self._full_text

        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            self._apply_elision()

        def _apply_elision(self) -> None:
            available_width = self.contentsRect().width()
            if available_width <= 0:
                super().setText(self._full_text)
                return
            text = self.fontMetrics().elidedText(
                self._full_text,
                QtCore.Qt.TextElideMode.ElideRight,
                available_width,
            )
            super().setText(text)

    class TastingWindow(QtWidgets.QMainWindow):
        def __init__(self, *, theme_id: str | None = None, preview_fixture: bool = False) -> None:
            super().__init__()
            self._theme = resolve_gui_theme(theme_id)
            self._preview_fixture = preview_fixture
            self._thread: QtCore.QThread | None = None
            self._worker: _FunctionWorker | None = None
            self._active_task_cancel_event: threading.Event | None = None
            if preview_fixture:
                self._runtime_status = GuiRuntimeStatus(
                    ready=True,
                    headline="Ready",
                    detail="",
                    missing_items=(),
                )
            else:
                self._runtime_status = probe_runtime_status()
            self._screen_state = GuiScreenState(runtime_status=self._runtime_status)
            self._output_srt_text = ""
            self._output_srt_user_override = False
            self._last_convert_result: ConvertResult | None = None
            self._check_feedback = _CheckFeedback()
            self._run_feedback = _RunFeedback()
            self._active_task_title = ""
            self._closing_anyway = False
            self._convert_progress_estimator: ConvertProgressEstimator | None = None
            self._setup_progress_started_at: float | None = None
            self._setup_progress_last_event_at: float | None = None
            self._progress_timer = QtCore.QTimer(self)
            self._progress_timer.setInterval(250)
            self._progress_timer.timeout.connect(self._refresh_convert_progress_display)
            self._setup_progress_timer = QtCore.QTimer(self)
            self._setup_progress_timer.setInterval(1000)
            self._setup_progress_timer.timeout.connect(self._refresh_setup_progress_display)

            self.setObjectName("AppWindow")
            self.setWindowTitle(f"istots {__version__}")
            self._apply_window_icon()
            self.resize(920, 610)
            self.setMinimumSize(760, 610)
            self._configure_palette()
            self._build_ui()
            self._apply_card_effects()
            if preview_fixture:
                self._load_preview_fixture()
            self._refresh_ui()
            self._apply_window_fit()

        def _apply_window_icon(self) -> None:
            icon = _build_gui_icon()
            if icon is None or icon.isNull():
                return
            app = QtWidgets.QApplication.instance()
            if app is not None and app.windowIcon().isNull():
                app.setWindowIcon(icon)
            self.setWindowIcon(icon)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            if self._thread is not None:
                title = self._active_task_title or "Task"
                answer = self._show_message_box(
                    icon=QtWidgets.QMessageBox.Icon.Warning,
                    title="Close App?",
                    message=(
                        f"{title} is still running.\n\n"
                        "Do you want to close now?"
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
                if answer != QtWidgets.QMessageBox.StandardButton.Close:
                    event.ignore()
                    return
                self._force_close_active_task_for_exit()
                self.hide()
                event.ignore()
                return
            self._cleanup_runtime_for_exit()
            super().closeEvent(event)

        def _configure_palette(self) -> None:
            theme = self._theme
            self.setStyleSheet(
                f"""
                QMainWindow#AppWindow {{
                    background: {theme.app_background};
                }}
                QWidget {{
                    background: transparent;
                    color: {theme.text_color};
                    font-size: {theme.base_font_size}px;
                }}
                QWidget#AppRoot {{
                    background: {theme.app_background};
                }}
                QFrame#FieldCard {{
                    background: {theme.card_background};
                    border: 1px solid {theme.card_border};
                    border-radius: {theme.card_radius}px;
                }}
                QFrame#StatusCard {{
                    background: {theme.status_card_background};
                    border: 1px solid {theme.card_border};
                    border-radius: {theme.card_radius}px;
                }}
                QFrame#ActionCard {{
                    background: {theme.action_card_background};
                    border: 1px solid {theme.card_border};
                    border-radius: {theme.card_radius}px;
                }}
                QFrame#SetupLane, QFrame#TestLane {{
                    background: {theme.input_background};
                    border: 1px solid {theme.card_border};
                    border-radius: {theme.input_radius}px;
                }}
                QLabel#LaneTitle, QLabel#ProgressTime {{
                    color: {theme.heading_color};
                }}
                QLabel#LaneTitle {{
                    font-size: {theme.base_font_size}px;
                    font-weight: 700;
                }}
                QLabel#SetupSummary, QLabel#TestSummary {{
                    color: {theme.muted_text};
                }}
                QFrame#StatusDivider {{
                    min-height: 1px;
                    max-height: 1px;
                    background: {theme.divider};
                    border: 0;
                }}
                QLineEdit {{
                    background: {theme.input_background};
                    border: 1px solid {theme.input_border};
                    border-radius: {theme.input_radius}px;
                    padding: 15px 16px;
                    selection-background-color: {theme.selection_background};
                }}
                QLineEdit:focus {{
                    border: 1px solid {theme.input_focus};
                }}
                QComboBox {{
                    background: {theme.input_background};
                    border: 1px solid {theme.input_border};
                    border-radius: {theme.input_radius}px;
                    padding: 10px 14px;
                    min-width: 138px;
                    selection-background-color: {theme.selection_background};
                }}
                QComboBox:focus {{
                    border: 1px solid {theme.input_focus};
                }}
                QComboBox::drop-down {{
                    border: 0;
                    width: 26px;
                }}
                QPushButton {{
                    border: 0;
                    border-radius: {theme.button_radius}px;
                    padding: 12px 18px;
                    font-weight: 700;
                    background: {theme.secondary_button_background};
                    color: {theme.secondary_button_text};
                }}
                QPushButton:hover {{
                    background: {theme.secondary_button_hover};
                }}
                QPushButton:disabled {{
                    background: {theme.secondary_button_background};
                    color: {theme.muted_text};
                }}
                QPushButton#IconButton {{
                    min-width: {theme.icon_button_size}px;
                    max-width: {theme.icon_button_size}px;
                    min-height: {theme.icon_button_size}px;
                    max-height: {theme.icon_button_size}px;
                    padding: 0;
                }}
                QPushButton#PrimaryButton {{
                    background: {theme.primary_background};
                    color: #fffaf4;
                    font-size: {theme.primary_font_size}px;
                    padding: 18px 24px;
                }}
                QPushButton#PrimaryButton:hover {{
                    background: {theme.primary_hover};
                }}
                QPushButton#PrimaryButton:disabled {{
                    background: {theme.primary_disabled};
                    color: {theme.primary_disabled_text};
                }}
                QLabel#ProgressDetail {{
                    color: {theme.heading_color};
                    font-size: {theme.progress_font_size}px;
                    font-weight: 700;
                }}
                QLabel#ProgressTime {{
                    font-size: {theme.time_font_size}px;
                }}
                QProgressBar {{
                    border: 0;
                    border-radius: {theme.progress_height // 2}px;
                    background: {theme.progress_background};
                    min-height: {theme.progress_height}px;
                    max-height: {theme.progress_height}px;
                }}
                QProgressBar[progressState="running"]::chunk {{
                    border-radius: {theme.progress_height // 2}px;
                    background: {theme.progress_chunk};
                }}
                QProgressBar[progressState="done"]::chunk {{
                    border-radius: {theme.progress_height // 2}px;
                    background: #1f7a49;
                }}
                QProgressBar[progressState="failed"]::chunk {{
                    border-radius: {theme.progress_height // 2}px;
                    background: #b94a2f;
                }}
                QMenu {{
                    background: {theme.card_background};
                    color: {theme.text_color};
                    border: 1px solid {theme.card_border};
                    padding: 6px;
                }}
                QMenu::item {{
                    padding: 8px 16px;
                    border-radius: 8px;
                    background: transparent;
                }}
                QMenu::item:selected {{
                    background: {theme.secondary_button_hover};
                    color: {theme.heading_color};
                }}
                QMenu::item:disabled {{
                    color: {theme.muted_text};
                }}
                QMenu::separator {{
                    height: 1px;
                    margin: 4px 8px;
                    background: {theme.divider};
                }}
                """
            )

        def _build_ui(self) -> None:
            theme = self._theme
            central = QtWidgets.QWidget(self)
            central.setObjectName("AppRoot")
            self._outer_layout = QtWidgets.QVBoxLayout(central)
            self._outer_layout.setContentsMargins(
                theme.outer_margin,
                theme.outer_margin,
                theme.outer_margin,
                theme.outer_margin,
            )
            self._outer_layout.setSpacing(theme.outer_spacing)
            self.setCentralWidget(central)

            self.status_card = QtWidgets.QFrame(objectName="StatusCard")
            status_policy = self.status_card.sizePolicy()
            status_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
            self.status_card.setSizePolicy(status_policy)
            self._status_layout = QtWidgets.QVBoxLayout(self.status_card)
            self._set_layout_margins(self._status_layout, theme.status_padding)
            self._status_layout.setSpacing(max(8, theme.card_spacing - 4))

            self.setup_lane = QtWidgets.QFrame(objectName="SetupLane")
            setup_lane_layout = QtWidgets.QHBoxLayout(self.setup_lane)
            self._set_layout_margins(setup_lane_layout, (14, 12, 14, 12))
            setup_lane_layout.setSpacing(10)
            self.ready_dot = _StatusGlyph("setup")
            self.setup_label = QtWidgets.QLabel("Setup", objectName="LaneTitle")
            self.setup_summary = _ElidedLabel(objectName="SetupSummary")
            setup_lane_layout.addWidget(self.ready_dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            setup_lane_layout.addWidget(self.setup_label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            setup_lane_layout.addWidget(self.setup_summary, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)

            divider = QtWidgets.QFrame(objectName="StatusDivider")
            divider.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

            self.setup_button = QtWidgets.QPushButton()
            self.setup_button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.setup_button.clicked.connect(self._handle_setup_action)
            self.runtime_variant_combo = QtWidgets.QComboBox()
            self.runtime_variant_combo.setToolTip("Setup runtime target")
            self.runtime_variant_combo.setAccessibleName("Runtime target")
            self.runtime_variant_combo.addItem("Auto", "auto")
            for variant_id in MANUAL_RUNTIME_VARIANTS:
                self.runtime_variant_combo.addItem(variant_id, variant_id)
            self.runtime_variant_combo.currentIndexChanged.connect(self._on_runtime_variant_changed)
            self.runtime_variant_combo.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            if os.name != "nt":
                self.runtime_variant_combo.hide()
            setup_lane_layout.addWidget(self.runtime_variant_combo, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            setup_lane_layout.addWidget(self.setup_button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

            self.test_lane = QtWidgets.QFrame(objectName="TestLane")
            test_lane_layout = QtWidgets.QHBoxLayout(self.test_lane)
            self._set_layout_margins(test_lane_layout, (14, 12, 14, 12))
            test_lane_layout.setSpacing(10)
            self.check_dot = _StatusGlyph("idle")
            self.test_label = QtWidgets.QLabel("Test", objectName="LaneTitle")
            self.test_summary = _ElidedLabel(objectName="TestSummary")
            self.refresh_button = QtWidgets.QPushButton("Test")
            self.refresh_button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.refresh_button.setFixedWidth(self.refresh_button.sizeHint().width())
            self.refresh_button.clicked.connect(self._start_runtime_check)
            test_lane_layout.addWidget(self.check_dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            test_lane_layout.addWidget(self.test_label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            test_lane_layout.addWidget(self.test_summary, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)
            test_lane_layout.addWidget(self.refresh_button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

            self._status_layout.addWidget(self.setup_lane)
            self._status_layout.addWidget(divider)
            self._status_layout.addWidget(self.test_lane)
            self._outer_layout.addWidget(self.status_card)

            self.fields_card = QtWidgets.QFrame(objectName="FieldCard")
            fields_policy = self.fields_card.sizePolicy()
            fields_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
            self.fields_card.setSizePolicy(fields_policy)
            self._fields_layout = QtWidgets.QVBoxLayout(self.fields_card)
            self._set_layout_margins(self._fields_layout, theme.fields_padding)
            self._fields_layout.setSpacing(theme.field_spacing)
            line_edit_height = max(theme.icon_button_size + 10, self.fontMetrics().height() + 26)

            input_row = QtWidgets.QHBoxLayout()
            input_row.setSpacing(12)
            self.input_edit = QtWidgets.QLineEdit()
            self.input_edit.setPlaceholderText("input.sup")
            self.input_edit.setReadOnly(True)
            self.input_edit.setFixedHeight(line_edit_height)
            self.input_browse = QtWidgets.QPushButton()
            self.input_browse.setObjectName("IconButton")
            self.input_browse.setAccessibleName("SUP")
            self.input_browse.setToolTip("SUP")
            self.input_browse.setFixedSize(theme.icon_button_size, theme.icon_button_size)
            self.input_browse.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
            )
            self.input_browse.setIconSize(QtCore.QSize(20, 20))
            self.input_browse.clicked.connect(self._choose_input_sup)
            input_row.addWidget(self.input_edit, 1)
            input_row.addWidget(self.input_browse)
            self._fields_layout.addLayout(input_row)

            output_row = QtWidgets.QHBoxLayout()
            output_row.setSpacing(12)
            self.output_edit = QtWidgets.QLineEdit()
            self.output_edit.setPlaceholderText("output.srt")
            self.output_edit.textEdited.connect(self._sync_output_text_state)
            self.output_edit.setFixedHeight(line_edit_height)
            self.output_browse = QtWidgets.QPushButton()
            self.output_browse.setObjectName("IconButton")
            self.output_browse.setAccessibleName("SRT")
            self.output_browse.setToolTip("SRT")
            self.output_browse.setFixedSize(theme.icon_button_size, theme.icon_button_size)
            self.output_browse.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)
            )
            self.output_browse.setIconSize(QtCore.QSize(20, 20))
            self.output_browse.clicked.connect(self._choose_output_srt)
            output_row.addWidget(self.output_edit, 1)
            output_row.addWidget(self.output_browse)
            self._fields_layout.addLayout(output_row)

            self.furigana_checkbox = _MaskCheckBox("Mask Furigana", theme=theme)
            self.furigana_checkbox.stateChanged.connect(self._sync_checkbox_state)
            self._fields_layout.addWidget(self.furigana_checkbox)
            self._outer_layout.addWidget(self.fields_card)

            self.action_card = QtWidgets.QFrame(objectName="ActionCard")
            action_policy = self.action_card.sizePolicy()
            action_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
            self.action_card.setSizePolicy(action_policy)
            self._action_layout = QtWidgets.QVBoxLayout(self.action_card)
            self._set_layout_margins(self._action_layout, theme.action_padding)
            self._action_layout.setSpacing(theme.action_spacing)
            self._action_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

            self.primary_button = QtWidgets.QPushButton(objectName="PrimaryButton")
            self.primary_button.setObjectName("PrimaryButton")
            self.primary_button.setMinimumHeight(max(56, self.primary_button.sizeHint().height()))
            self.primary_button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.primary_button.clicked.connect(self._handle_primary_action)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setTextVisible(False)
            self.progress.setRange(0, 0)
            self.progress.setProperty("progressState", "running")
            self.progress.hide()
            self.progress_detail = QtWidgets.QLabel(objectName="ProgressDetail")
            self.progress_time = QtWidgets.QLabel(objectName="ProgressTime")
            self.progress_detail.hide()
            self.progress_time.hide()
            progress_detail_height = max(
                self.progress_detail.sizeHint().height(),
                QtGui.QFontMetrics(self.progress_detail.font()).height(),
            )
            progress_time_height = max(
                self.progress_time.sizeHint().height(),
                QtGui.QFontMetrics(self.progress_time.font()).height(),
            )
            reserved_action_height = (
                self.primary_button.minimumHeight()
                + theme.progress_height
                + progress_detail_height
                + progress_time_height
                + (theme.action_spacing * 2)
                + theme.action_padding[1]
                + theme.action_padding[3]
            )
            self.action_card.setMinimumHeight(reserved_action_height)

            self._action_layout.addWidget(self.primary_button)
            self._action_layout.addWidget(self.progress)
            self._action_layout.addWidget(self.progress_detail)
            self._action_layout.addWidget(self.progress_time)
            self._outer_layout.addWidget(self.action_card, 1)

        def _set_layout_margins(
            self,
            layout: QtWidgets.QLayout,
            margins: tuple[int, int, int, int],
        ) -> None:
            left, top, right, bottom = margins
            layout.setContentsMargins(left, top, right, bottom)

        def _apply_window_fit(self) -> None:
            self._ensure_window_fit(min_height=610, max_height=660, expand_only=False)

        def _ensure_window_fit(
            self,
            *,
            min_height: int,
            max_height: int,
            expand_only: bool,
        ) -> None:
            central = self.centralWidget()
            if central is None:
                return
            content_height = central.sizeHint().height()
            target_height = max(min_height, min(max_height, content_height + 24))
            if expand_only and self.height() >= target_height:
                return
            self.resize(self.width(), target_height)

        def _apply_card_effects(self) -> None:
            theme = self._theme
            for card in (self.status_card, self.fields_card, self.action_card):
                if theme.shadow_blur <= 0:
                    card.setGraphicsEffect(None)
                    continue
                effect = QtWidgets.QGraphicsDropShadowEffect(card)
                effect.setBlurRadius(theme.shadow_blur)
                effect.setOffset(0, theme.shadow_offset_y)
                effect.setColor(QtGui.QColor(theme.shadow_color))
                card.setGraphicsEffect(effect)

        def _load_preview_fixture(self) -> None:
            self._runtime_status = GuiRuntimeStatus(
                ready=True,
                headline="Ready",
                detail="",
                missing_items=(),
            )
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=Path("test.sup"),
                output_srt=Path("test.srt"),
                enable_furigana_mask=True,
            )
            self._output_srt_text = "test.srt"
            self._output_srt_user_override = False
            self._check_feedback = _CheckFeedback(state="ok", summary="Passed", detail="Runtime test passed.")
            self._set_run_feedback(
                state="running",
                detail="OCR 322/518 62%",
                time_text="12:48 / est. 08:15 left",
                value=620,
            )

        def _set_progress_state(self, state: str) -> None:
            self.progress.setProperty("progressState", state)
            style = self.progress.style()
            style.unpolish(self.progress)
            style.polish(self.progress)
            self.progress.update()

        def _set_run_feedback(
            self,
            *,
            state: str,
            detail: str,
            time_text: str,
            value: int,
            visible: bool = True,
        ) -> None:
            clamped = max(0, min(1000, value))
            self._run_feedback = _RunFeedback(
                state=state,
                detail=detail,
                time_text=time_text,
                value=clamped,
                visible=visible,
            )
            self._apply_run_feedback()

        def _clear_run_feedback(self) -> None:
            self._run_feedback = _RunFeedback()
            self._apply_run_feedback()

        def _clear_setup_progress(self) -> None:
            self._setup_progress_timer.stop()
            self._setup_progress_started_at = None
            self._setup_progress_last_event_at = None
            self.progress.hide()
            self.progress_detail.clear()
            self.progress_detail.setToolTip("")
            self.progress_detail.hide()
            self.progress_time.clear()
            self.progress_time.hide()
            self._set_progress_state("running")

        def _apply_run_feedback(self) -> None:
            feedback = self._run_feedback
            if not feedback.visible:
                self.progress.hide()
                self.progress_detail.hide()
                self.progress_time.hide()
                return

            self.progress.setRange(0, 1000)
            self.progress.setValue(feedback.value)
            self._set_progress_state(feedback.state)
            self.progress.show()

            self.progress_detail.setText(feedback.detail)
            self.progress_detail.setVisible(bool(feedback.detail))
            self.progress_time.setText(feedback.time_text)
            self.progress_time.setVisible(bool(feedback.time_text))

        def _probe_runtime_status(self) -> None:
            self._runtime_status = probe_runtime_status()
            if not self._runtime_status.ready:
                self._set_check_feedback("idle", "")
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=self._screen_state.input_sup,
                output_srt=self._screen_state.output_srt,
                enable_furigana_mask=self._screen_state.enable_furigana_mask,
            )
            self._refresh_ui()

        def _selected_runtime_variant(self) -> str:
            value = self.runtime_variant_combo.currentData()
            return str(value) if value else "auto"

        def _on_runtime_variant_changed(self) -> None:
            self._apply_runtime_facts(self._runtime_status)

        def _start_runtime_check(self) -> None:
            self._set_check_feedback("busy", "")
            cancel_event = threading.Event()
            self._start_task(
                title="Check",
                fn=lambda cancel_event=cancel_event: run_gui_doctor_check(cancel_event=cancel_event),
                on_success=self._on_runtime_check_finished,
                cancel_event=cancel_event,
            )

        def _sync_checkbox_state(self) -> None:
            if self._active_task_title != "Run":
                self._clear_run_feedback()
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=self._screen_state.input_sup,
                output_srt=self._screen_state.output_srt,
                enable_furigana_mask=self.furigana_checkbox.isChecked(),
            )
            self._refresh_ui()

        def _sync_output_text_state(self, text: str) -> None:
            if self._active_task_title != "Run":
                self._clear_run_feedback()
            self._output_srt_text = text
            self._output_srt_user_override = True
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=self._screen_state.input_sup,
                output_srt=None if text == "" else Path(text),
                enable_furigana_mask=self.furigana_checkbox.isChecked(),
            )
            self._refresh_ui()

        def _choose_input_sup(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "SUP",
                "",
                "SUP subtitles (*.sup)",
            )
            if not path:
                return
            self._clear_run_feedback()
            input_path = Path(path).expanduser().resolve()
            if self._output_srt_user_override:
                output_path = self._screen_state.output_srt
            else:
                output_path = suggest_output_srt_path(input_path)
                self._output_srt_text = str(output_path)
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=input_path,
                output_srt=output_path,
                enable_furigana_mask=self.furigana_checkbox.isChecked(),
            )
            self._refresh_ui()

        def _choose_output_srt(self) -> None:
            suggested = ""
            if self._screen_state.output_srt is not None:
                suggested = str(self._screen_state.output_srt)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "SRT",
                suggested,
                "SRT subtitles (*.srt)",
            )
            if not path:
                return
            self._clear_run_feedback()
            output_path = Path(path).expanduser().resolve()
            self._output_srt_text = str(output_path)
            self._output_srt_user_override = True
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=self._screen_state.input_sup,
                output_srt=output_path,
                enable_furigana_mask=self.furigana_checkbox.isChecked(),
            )
            self._refresh_ui()

        def _handle_setup_action(self) -> None:
            action = derive_setup_action(self._screen_state)
            if not action.enabled:
                return
            install_prerequisites = self._confirm_setup_prerequisites()
            if install_prerequisites is None:
                return
            cancel_event = threading.Event()
            self._start_task(
                title="Setup",
                fn=lambda emit, install_prerequisites=install_prerequisites, cancel_event=cancel_event: self._run_setup_task(
                    emit,
                    install_prerequisites=install_prerequisites,
                    cancel_event=cancel_event,
                ),
                on_success=self._on_setup_finished,
                on_progress=self._on_setup_progress_event,
                cancel_event=cancel_event,
            )

        def _confirm_setup_prerequisites(self) -> bool | None:
            if os.name != "nt":
                return False
            missing = missing_managed_runtime_prerequisites()
            if not missing:
                return False
            answer = self._show_message_box(
                icon=QtWidgets.QMessageBox.Icon.Warning,
                title="One More Step",
                message=(
                    "Windows needs Microsoft Visual C++ (x64) for this runtime.\n\n"
                    + format_missing_managed_runtime_prerequisites(missing)
                ),
                buttons=QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
                default_button=QtWidgets.QMessageBox.StandardButton.Yes,
                button_text_overrides={
                    QtWidgets.QMessageBox.StandardButton.Yes: "Install Now",
                    QtWidgets.QMessageBox.StandardButton.Cancel: "Not Now",
                },
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return None
            return True

        def _run_setup_task(
            self,
            emit,
            *,
            install_prerequisites: bool = False,
            cancel_event: threading.Event | None = None,
        ) -> SetupResult:
            request = build_setup_request_for_variant(
                runtime_variant=self._selected_runtime_variant(),
                install_prerequisites=install_prerequisites,
            )
            append_runtime_diagnostic_event(
                "gui_setup_task_start",
                runtime_variant=request.runtime_variant,
                install_prerequisites=install_prerequisites,
                models_dir=request.models_dir,
            )
            try:
                result = execute_setup_request(
                    request,
                    progress_callback=emit,
                    emit_completion_event=False,
                    cancel_event=cancel_event,
                )
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("setup cancelled during runtime validation")
                emit(
                    SetupProgressEvent(
                        phase="runtime_check",
                        headline="Validate Setup",
                        detail="Waiting for llama-server to respond",
                        fraction=0.97,
                    )
                )
                append_runtime_diagnostic_event(
                    "gui_setup_doctor_start",
                    models_dir=request.models_dir,
                )
                status = run_gui_doctor_check(models_dir=request.models_dir, cancel_event=cancel_event)
                append_runtime_diagnostic_event(
                    "gui_setup_doctor_complete",
                    ready=status.ready,
                    detail=status.detail,
                    runtime_source=status.runtime_source,
                    runtime_binary_path=status.runtime_binary_path,
                )
                if not status.ready:
                    if status.runtime_source == "managed":
                        record_managed_runtime_validation(
                            ok=False,
                            detail=status.detail,
                            binary_path=status.runtime_binary_path,
                        )
                    raise RuntimeError(status.detail)
                if status.runtime_source == "managed":
                    record_managed_runtime_validation(
                        ok=True,
                        detail=status.detail,
                        binary_path=status.runtime_binary_path,
                    )
                emit(
                    SetupProgressEvent(
                        phase="complete",
                        headline="Setup Complete",
                        detail="Runtime test passed",
                        fraction=1.0,
                    )
                )
                append_runtime_diagnostic_event("gui_setup_task_complete")
                return result
            except Exception as exc:
                append_runtime_diagnostic_event(
                    "gui_setup_task_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise

        def _handle_primary_action(self) -> None:
            action = derive_primary_action(self._screen_state)
            if not action.enabled:
                return
            try:
                plan = self._prepare_convert_plan()
            except (ConvertArgumentError, ConvertPreparationError, RuntimeError) as exc:
                self._show_message_box(
                    icon=QtWidgets.QMessageBox.Icon.Critical,
                    title="Run",
                    message=str(exc),
                )
                return
            if not self._confirm_overwrite(plan.existing_output_artifacts):
                return
            self._begin_convert_progress(plan)
            cancel_event = threading.Event()
            self._start_task(
                title="Run",
                fn=lambda emit, plan=plan, cancel_event=cancel_event: execute_convert_plan(
                    plan,
                    verbose=False,
                    progress_callback=emit,
                    cancel_event=cancel_event,
                ),
                on_success=self._on_convert_finished,
                on_progress=self._on_convert_progress_event,
                cancel_event=cancel_event,
            )

        def _prepare_convert_plan(self):
            if self._screen_state.input_sup is None or self._screen_state.output_srt is None:
                raise RuntimeError("SUP")
            request = build_fast_convert_request(
                input_sup=self._screen_state.input_sup,
                output_srt=self._screen_state.output_srt,
                enable_furigana_mask=self._screen_state.enable_furigana_mask,
                runtime_status=self._runtime_status,
            )
            return plan_convert_request(request)

        def _confirm_overwrite(self, existing_paths: tuple[Path, ...]) -> bool:
            if not existing_paths:
                return True

            if len(existing_paths) == 1:
                prompt = f"This file already exists.\n\n{existing_paths[0].name}"
            else:
                prompt = "These files already exist.\n\n" + "\n".join(path.name for path in existing_paths)

            answer = self._show_message_box(
                icon=QtWidgets.QMessageBox.Icon.Warning,
                title="Replace File?",
                message=prompt,
                buttons=QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                default_button=QtWidgets.QMessageBox.StandardButton.No,
                button_text_overrides={
                    QtWidgets.QMessageBox.StandardButton.Yes: "Replace",
                    QtWidgets.QMessageBox.StandardButton.No: "Keep",
                },
            )
            return answer == QtWidgets.QMessageBox.StandardButton.Yes

        def _begin_convert_progress(self, plan) -> None:
            self._convert_progress_estimator = ConvertProgressEstimator(
                input_sup=plan.input_sup,
                enable_furigana_mask=plan.enable_furigana_mask,
                ocr_mode=plan.ocr_mode,
            )
            self._set_run_feedback(state="running", detail="", time_text="", value=0)
            self._refresh_convert_progress_display()

        def _clear_convert_progress(self) -> None:
            self._progress_timer.stop()
            self._convert_progress_estimator = None

        def _start_task(self, *, title: str, fn, on_success, on_progress=None, cancel_event=None) -> None:
            if self._thread is not None:
                return
            self._active_task_title = title
            self._active_task_cancel_event = cancel_event
            if title == "Run":
                self._progress_timer.start()
                self._refresh_convert_progress_display()
            elif title == "Setup":
                self._clear_run_feedback()
                self._set_progress_state("running")
                self.progress.show()
                self.progress.setRange(0, 1000)
                self.progress.setValue(0)
                self.progress_detail.setText("Preparing setup")
                self.progress_detail.setToolTip("")
                self.progress_detail.show()
                self._begin_setup_progress()
            else:
                self.progress.show()
                self.progress.setRange(0, 0)
                self.progress_detail.hide()
                self.progress_time.hide()
            self._set_busy(True)

            self._thread = QtCore.QThread(self)
            worker: _FunctionWorker
            if on_progress is None:
                worker = _FunctionWorker(title, fn)
            else:
                def _run_with_progress():
                    return fn(worker.progressed.emit)

                worker = _FunctionWorker(title, _run_with_progress)
                worker.progressed.connect(on_progress)
            self._worker = worker
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.succeeded.connect(on_success)
            self._worker.failed.connect(self._on_task_failed)
            self._worker.finished.connect(self._thread.quit)
            self._worker.finished.connect(self._worker.deleteLater)
            self._thread.finished.connect(self._thread.deleteLater)
            self._thread.finished.connect(self._on_task_finished)
            self._thread.start()

        def _set_busy(self, busy: bool) -> None:
            self.primary_button.setDisabled(busy)
            self.setup_button.setDisabled(busy)
            self.runtime_variant_combo.setDisabled(busy)
            self.input_edit.setDisabled(busy)
            self.output_edit.setDisabled(busy)
            self.furigana_checkbox.setDisabled(busy)
            if busy:
                self.refresh_button.setDisabled(True)
            else:
                self.refresh_button.setDisabled(not self._runtime_status.ready)

        def _on_task_finished(self) -> None:
            finished_title = self._active_task_title
            if finished_title == "Run":
                self._clear_convert_progress()
                self._apply_run_feedback()
            else:
                self._clear_setup_progress()
                if self._run_feedback.visible:
                    self._apply_run_feedback()
            self._active_task_title = ""
            self._thread = None
            self._worker = None
            self._active_task_cancel_event = None
            self._set_busy(False)
            self._refresh_ui()
            if self._closing_anyway:
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    QtCore.QTimer.singleShot(0, app.quit)

        def _on_task_failed(self, title: str, message: str) -> None:
            if title == "Check":
                self._set_check_feedback("fail", message)
            elif title == "Setup":
                status = probe_runtime_status()
                self._runtime_status = replace(
                    status,
                    ready=False,
                    headline="Setup",
                    detail=message,
                    missing_items=status.missing_items or ("runtime-validation",),
                )
                self._screen_state = GuiScreenState(
                    runtime_status=self._runtime_status,
                    input_sup=self._screen_state.input_sup,
                    output_srt=self._screen_state.output_srt,
                    enable_furigana_mask=self._screen_state.enable_furigana_mask,
                )
                self._set_check_feedback("fail", message)
                self._clear_setup_progress()
            elif title == "Run":
                elapsed = ""
                value = 0
                if self._convert_progress_estimator is not None:
                    snapshot = self._convert_progress_estimator.snapshot()
                    elapsed = self._format_duration(snapshot.elapsed_sec)
                    value = int(round(snapshot.fraction * 1000))
                self._set_run_feedback(
                    state="failed",
                    detail="Failed",
                    time_text=elapsed,
                    value=value,
                )
            if not self._closing_anyway:
                self._show_message_box(
                    icon=QtWidgets.QMessageBox.Icon.Critical,
                    title=title,
                    message=message,
                )

        def _cleanup_runtime_for_exit(self) -> None:
            self._closing_anyway = True
            cancel_event = self._active_task_cancel_event
            if cancel_event is not None:
                cancel_event.set()
            try:
                from istots import llama_runtime

                llama_runtime.request_llama_server_process_shutdown()
                llama_runtime.cleanup_managed_llama_server_for_current_process()
            except Exception:
                pass

        def _terminate_active_task_thread_if_still_running(self) -> None:
            if not self._closing_anyway:
                return
            thread = self._thread
            if thread is None:
                return
            is_finished = getattr(thread, "isFinished", None)
            if callable(is_finished) and is_finished():
                return
            thread.terminate()
            thread.wait(1000)

        def _force_close_active_task_for_exit(self) -> None:
            self._cleanup_runtime_for_exit()
            self._clear_convert_progress()
            self._clear_setup_progress()

            thread = self._thread
            if thread is None:
                return
            thread.requestInterruption()
            thread.quit()
            if thread.wait(250):
                return
            QtCore.QTimer.singleShot(1500, self._terminate_active_task_thread_if_still_running)

        def _on_setup_progress_event(self, event: SetupProgressEvent) -> None:
            self._record_setup_progress_event()
            self._set_progress_state("running")
            self.progress.show()
            if event.fraction is None:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 1000)
                self.progress.setValue(max(0, min(1000, int(round(event.fraction * 1000)))))
            detail, full_detail = _format_setup_progress_label_text(event)
            self.progress_detail.setText(detail)
            self.progress_detail.setToolTip("" if not full_detail or full_detail == detail else full_detail)
            self.progress_detail.setVisible(bool(detail))
            self._refresh_setup_progress_display()

        def _on_setup_finished(self, _result: SetupResult) -> None:
            self._runtime_status = probe_runtime_status()
            self._set_check_feedback("idle", "")
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=self._screen_state.input_sup,
                output_srt=self._screen_state.output_srt,
                enable_furigana_mask=self._screen_state.enable_furigana_mask,
            )
            self._refresh_ui()

        def _on_runtime_check_finished(self, status: GuiRuntimeStatus) -> None:
            self._runtime_status = probe_runtime_status()
            if status.ready:
                if status.runtime_source == "managed":
                    record_managed_runtime_validation(
                        ok=True,
                        detail=status.detail,
                        binary_path=status.runtime_binary_path,
                    )
                self._set_check_feedback("ok", status.detail)
            else:
                if status.runtime_source == "managed":
                    record_managed_runtime_validation(
                        ok=False,
                        detail=status.detail,
                        binary_path=status.runtime_binary_path,
                    )
                self._set_check_feedback("fail", status.detail)
                self._show_message_box(
                    icon=QtWidgets.QMessageBox.Icon.Warning,
                    title="Test",
                    message=status.detail,
                )
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=self._screen_state.input_sup,
                output_srt=self._screen_state.output_srt,
                enable_furigana_mask=self._screen_state.enable_furigana_mask,
            )
            self._refresh_ui()

        def _on_convert_finished(self, result: ConvertResult) -> None:
            self._last_convert_result = result
            elapsed = ""
            if self._convert_progress_estimator is not None:
                snapshot = self._convert_progress_estimator.snapshot()
                elapsed = f"{self._format_duration(snapshot.elapsed_sec)} total"
            self._set_run_feedback(
                state="done",
                detail="Done",
                time_text=elapsed,
                value=1000,
            )
            self._runtime_status = probe_runtime_status()
            self._screen_state = GuiScreenState(
                runtime_status=self._runtime_status,
                input_sup=self._screen_state.input_sup,
                output_srt=result.output_srt,
                enable_furigana_mask=self._screen_state.enable_furigana_mask,
            )
            self._output_srt_text = str(result.output_srt)
            self._refresh_ui()

        def _on_convert_progress_event(self, event: ConvertProgressEvent) -> None:
            if self._convert_progress_estimator is None:
                return
            self._convert_progress_estimator.record(event)
            self._refresh_convert_progress_display()

        def _refresh_convert_progress_display(self) -> None:
            if self._active_task_title != "Run" or self._convert_progress_estimator is None:
                return
            snapshot = self._convert_progress_estimator.snapshot()
            self._set_run_feedback(
                state="running",
                detail=self._format_progress_detail(snapshot),
                time_text=self._format_progress_time(snapshot),
                value=int(round(snapshot.fraction * 1000)),
            )

        def _begin_setup_progress(self) -> None:
            started_at = time.monotonic()
            self._setup_progress_started_at = started_at
            self._setup_progress_last_event_at = started_at
            self._setup_progress_timer.start()
            self._refresh_setup_progress_display()

        def _record_setup_progress_event(self) -> None:
            now = time.monotonic()
            if self._setup_progress_started_at is None:
                self._setup_progress_started_at = now
                self._setup_progress_timer.start()
            self._setup_progress_last_event_at = now

        def _refresh_setup_progress_display(self) -> None:
            if self._active_task_title != "Setup" or self._setup_progress_started_at is None:
                return
            time_text = self._format_setup_progress_time()
            self.progress_time.setText(time_text)
            self.progress_time.setVisible(bool(time_text))

        def _format_setup_progress_time(self) -> str:
            started_at = self._setup_progress_started_at
            if started_at is None:
                return ""
            now = time.monotonic()
            elapsed = self._format_duration(now - started_at)
            return f"Still working... {elapsed} elapsed"

        def _format_progress_detail(self, snapshot: ConvertProgressSnapshot) -> str:
            percent = int(round(snapshot.fraction * 100.0))
            if snapshot.detail:
                return f"{snapshot.headline} {snapshot.detail} {percent}%"
            return f"{snapshot.headline} {percent}%"

        def _format_progress_time(self, snapshot: ConvertProgressSnapshot) -> str:
            elapsed = self._format_duration(snapshot.elapsed_sec)
            if snapshot.eta_sec is None:
                return elapsed
            eta = self._format_duration(snapshot.eta_sec)
            return f"{elapsed} / est. {eta} left"

        def _format_duration(self, total_seconds: float) -> str:
            total = max(0, int(round(total_seconds)))
            hours, remainder = divmod(total, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            return f"{minutes:02d}:{seconds:02d}"

        def _set_dot_state(self, dot: _StatusGlyph, state: str) -> None:
            dot.set_state(state)

        def _set_check_feedback(self, state: str, detail: str) -> None:
            self._check_feedback = _CheckFeedback(
                state=state,
                summary=_default_check_summary(state),
                detail=detail,
            )

        def _refresh_ui(self) -> None:
            action = derive_primary_action(self._screen_state)
            setup_action = derive_setup_action(self._screen_state)
            self._apply_runtime_status(self._runtime_status)
            self._apply_check_feedback()
            self.primary_button.setText(action.label)
            self.primary_button.setEnabled(action.enabled)
            self.setup_button.setText(setup_action.label)
            self.setup_button.setEnabled(setup_action.enabled)
            desired_input_text = "" if self._screen_state.input_sup is None else str(self._screen_state.input_sup)
            if self.input_edit.text() != desired_input_text:
                self.input_edit.setText(desired_input_text)
            if self.output_edit.text() != self._output_srt_text:
                self.output_edit.setText(self._output_srt_text)
            checkbox_blocker = QtCore.QSignalBlocker(self.furigana_checkbox)
            try:
                self.furigana_checkbox.setChecked(self._screen_state.enable_furigana_mask)
            finally:
                del checkbox_blocker

        def _apply_runtime_status(self, status: GuiRuntimeStatus) -> None:
            self._set_dot_state(self.ready_dot, "ready" if status.ready else "setup")
            self._apply_runtime_facts(status)
            self.refresh_button.setDisabled(self._thread is not None or not status.ready)

        def _apply_runtime_facts(self, status: GuiRuntimeStatus) -> None:
            summary = format_setup_summary(
                status=status,
                selected_variant=self._selected_runtime_variant(),
            )
            facts = format_runtime_facts(
                status=status,
                selected_variant=self._selected_runtime_variant(),
            )
            tooltip_parts = [part for part in (facts, status.detail) if part]
            tooltip = "\n\n".join(tooltip_parts)
            self.setup_summary.set_full_text(summary, tooltip=tooltip)
            self.setup_lane.setToolTip(tooltip)
            self.setup_label.setToolTip(tooltip)

        def _apply_check_feedback(self) -> None:
            self._set_dot_state(self.check_dot, self._check_feedback.state)
            summary = format_check_summary(
                state=self._check_feedback.state,
                detail=self._check_feedback.detail,
            )
            tooltip = self._check_feedback.detail or summary
            self.test_summary.set_full_text(summary, tooltip=tooltip)
            self.test_lane.setToolTip(tooltip)
            self.test_label.setToolTip(tooltip)

        def _show_message_box(
            self,
            *,
            icon: QtWidgets.QMessageBox.Icon,
            title: str,
            message: str,
            buttons: QtWidgets.QMessageBox.StandardButton = QtWidgets.QMessageBox.StandardButton.Ok,
            default_button: QtWidgets.QMessageBox.StandardButton | None = None,
            button_text_overrides: dict[QtWidgets.QMessageBox.StandardButton, str] | None = None,
        ) -> QtWidgets.QMessageBox.StandardButton:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle(title)
            box.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            box.setObjectName("ThemedMessageBox")
            box.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
            primary_text, informative_text, detailed_text = _split_message_box_content(
                message,
                font=box.font(),
                max_width_px=_message_box_max_width(self, detailed=False),
            )
            max_text_width = _message_box_max_width(self, detailed=bool(detailed_text))
            box.setIcon(QtWidgets.QMessageBox.Icon.NoIcon)
            box.setText(primary_text or _wrap_message_box_text(message, font=box.font(), max_width_px=max_text_width))
            if informative_text:
                box.setInformativeText(informative_text)
            if detailed_text:
                box.setDetailedText(detailed_text)
            box.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            box.setStandardButtons(buttons)
            if button_text_overrides:
                for button_id, text in button_text_overrides.items():
                    button = box.button(button_id)
                    if button is not None:
                        button.setText(text)
            if default_button is not None:
                box.setDefaultButton(default_button)
            theme = self._theme
            box.setStyleSheet(
                f"""
                QMessageBox#ThemedMessageBox {{
                    background: {theme.card_background};
                    border: 1px solid {theme.card_border};
                }}
                QMessageBox#ThemedMessageBox QLabel {{
                    background: transparent;
                    color: {theme.text_color};
                }}
                QMessageBox#ThemedMessageBox QLabel#qt_msgbox_label {{
                    min-width: 0px;
                    max-width: {max_text_width}px;
                    color: {theme.text_color};
                    font-size: {theme.base_font_size}px;
                }}
                QMessageBox#ThemedMessageBox QLabel#qt_msgbox_informativelabel {{
                    min-width: 0px;
                    max-width: {max_text_width}px;
                    color: {theme.muted_text};
                    font-size: {theme.base_font_size}px;
                }}
                QMessageBox#ThemedMessageBox QLabel#qt_msgboxex_icon_label {{
                    min-width: 0px;
                    max-width: 0px;
                }}
                QMessageBox#ThemedMessageBox QPushButton {{
                    border: 0;
                    border-radius: {theme.button_radius}px;
                    padding: 10px 16px;
                    font-weight: 700;
                    background: {theme.secondary_button_background};
                    color: {theme.secondary_button_text};
                    min-width: 0px;
                }}
                QMessageBox#ThemedMessageBox QPushButton:hover {{
                    background: {theme.secondary_button_hover};
                }}
                QMessageBox#ThemedMessageBox QPushButton:default {{
                    background: {theme.primary_background};
                    color: {theme.primary_disabled_text};
                }}
                QMessageBox#ThemedMessageBox QTextEdit,
                QMessageBox#ThemedMessageBox QPlainTextEdit {{
                    background: {theme.input_background};
                    color: {theme.text_color};
                    border: 1px solid {theme.input_border};
                    border-radius: {theme.input_radius}px;
                    selection-background-color: {theme.selection_background};
                }}
                QMessageBox#ThemedMessageBox QMenu {{
                    background: {theme.card_background};
                    color: {theme.text_color};
                    border: 1px solid {theme.card_border};
                    padding: 6px;
                }}
                QMessageBox#ThemedMessageBox QMenu::item {{
                    padding: 8px 16px;
                    border-radius: 8px;
                    background: transparent;
                }}
                QMessageBox#ThemedMessageBox QMenu::item:selected {{
                    background: {theme.secondary_button_hover};
                    color: {theme.heading_color};
                }}
                """
            )
            box.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
            grid_layout = box.layout() if isinstance(box.layout(), QtWidgets.QGridLayout) else None
            if grid_layout is not None:
                grid_layout.setHorizontalSpacing(0)
                grid_layout.setColumnMinimumWidth(0, 0)
                grid_layout.setColumnMinimumWidth(1, 0)
            for name in ("qt_msgbox_label", "qt_msgbox_informativelabel"):
                label = box.findChild(QtWidgets.QLabel, name)
                if label is None:
                    continue
                label.setWordWrap(True)
                label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
                label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
                label.setMinimumWidth(min(420, max_text_width))
                label.setMaximumWidth(max_text_width)
                if grid_layout is not None:
                    grid_layout.removeWidget(label)
                    row = 0 if name == "qt_msgbox_label" else 1
                    grid_layout.addWidget(label, row, 0, 1, 2)
                label.adjustSize()
            icon_label = box.findChild(QtWidgets.QLabel, "qt_msgboxex_icon_label")
            if icon_label is not None:
                if grid_layout is not None:
                    grid_layout.removeWidget(icon_label)
                icon_label.hide()
            if grid_layout is not None:
                grid_layout.invalidate()
            for details_edit in box.findChildren(QtWidgets.QTextEdit):
                details_edit.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
                details_edit.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
                details_edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                details_edit.setMinimumWidth(max_text_width)
                details_edit.setMinimumHeight(180)
            for details_edit in box.findChildren(QtWidgets.QPlainTextEdit):
                details_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
                details_edit.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
                details_edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                details_edit.setMinimumWidth(max_text_width)
                details_edit.setMinimumHeight(180)
            _apply_message_box_button_widths(box)
            box.setMinimumWidth(max_text_width + 120)
            box.layout().activate()
            box.adjustSize()
            result = box.exec()
            return QtWidgets.QMessageBox.StandardButton(int(result))


def _compose_theme_sheet(
    image_paths: list[Path],
    *,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    labels = [resolve_gui_theme(path.stem).label for path in image_paths]
    images = [Image.open(path).convert("RGBA") for path in image_paths]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    gap = 28
    pad = 28
    label_height = 56
    total_width = pad * 2 + sum(image.width for image in images) + gap * (len(images) - 1)
    max_height = max(image.height for image in images)
    sheet = Image.new("RGBA", (total_width, pad * 2 + label_height + max_height), "#f6efe6")
    draw = ImageDraw.Draw(sheet)

    x = pad
    for image, label in zip(images, labels, strict=True):
        draw.rounded_rectangle(
            (x - 8, pad + label_height - 8, x + image.width + 8, pad + label_height + image.height + 8),
            radius=24,
            fill="#fffaf4",
            outline="#e7d5c4",
            width=2,
        )
        draw.text((x, pad), label, fill="#2b211c", font=font)
        sheet.paste(image, (x, pad + label_height), image)
        x += image.width + gap

    sheet.save(output_path)


def render_theme_previews(output_dir: Path) -> tuple[Path, ...]:
    _ensure_qt()
    assert QtWidgets is not None

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])
        _apply_application_metadata(app)
    else:
        _apply_application_metadata(app)

    rendered_paths: list[Path] = []
    try:
        for theme_id in list_gui_theme_ids():
            window = TastingWindow(theme_id=theme_id, preview_fixture=True)
            window.show()
            app.processEvents()
            app.processEvents()
            image_path = output_dir / f"{theme_id}.png"
            window.grab().save(str(image_path))
            rendered_paths.append(image_path)
            window.close()

        sheet_path = output_dir / "theme_compare_sheet.png"
        _compose_theme_sheet(rendered_paths, output_path=sheet_path)
        rendered_paths.append(sheet_path)
        return tuple(rendered_paths)
    finally:
        if owns_app:
            app.quit()


def launch_gui(*, theme_id: str | None = None) -> int:
    _ensure_qt()
    assert QtWidgets is not None
    from istots import llama_runtime

    llama_runtime.clear_llama_server_process_shutdown_request()
    app = QtWidgets.QApplication([])
    _apply_application_metadata(app)
    window = TastingWindow(theme_id=theme_id)
    app.aboutToQuit.connect(window._cleanup_runtime_for_exit)
    window.show()
    return app.exec()
