"""GUI entrypoints and contracts for the minimal tasting-shell frontend."""

from .core import (
    GuiPrimaryAction,
    GuiRuntimeStatus,
    GuiScreenState,
    build_fast_convert_request,
    build_setup_request,
    derive_primary_action,
    derive_setup_action,
    probe_runtime_status,
    run_gui_doctor_check,
    suggest_output_srt_path,
)

__all__ = [
    "GuiPrimaryAction",
    "GuiRuntimeStatus",
    "GuiScreenState",
    "build_fast_convert_request",
    "build_setup_request",
    "derive_primary_action",
    "derive_setup_action",
    "probe_runtime_status",
    "run_gui_doctor_check",
    "suggest_output_srt_path",
]
