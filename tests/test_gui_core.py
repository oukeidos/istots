from __future__ import annotations

import os
from pathlib import Path
import threading

import pytest

from istots.app.convert import ConvertRequest
from istots.app.setup import SetupRequest
from istots.gui.core import (
    GuiRuntimeStatus,
    GuiScreenState,
    build_fast_convert_request,
    build_setup_request,
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


def test_suggest_output_srt_path_swaps_suffix(tmp_path: Path) -> None:
    input_sup = tmp_path / "episode.sup"

    output_srt = suggest_output_srt_path(input_sup)

    assert output_srt == (tmp_path / "episode.srt").resolve()


def test_suggest_output_srt_path_avoids_existing_target(tmp_path: Path) -> None:
    input_sup = tmp_path / "episode.sup"
    (tmp_path / "episode.srt").write_text("old", encoding="utf-8")

    output_srt = suggest_output_srt_path(input_sup)

    assert output_srt == (tmp_path / "episode (2).srt").resolve()


def test_suggest_output_srt_path_skips_taken_numbered_targets(tmp_path: Path) -> None:
    input_sup = tmp_path / "episode.sup"
    (tmp_path / "episode.srt").write_text("old", encoding="utf-8")
    (tmp_path / "episode (2).srt").write_text("older", encoding="utf-8")

    output_srt = suggest_output_srt_path(input_sup)

    assert output_srt == (tmp_path / "episode (3).srt").resolve()


def test_derive_primary_action_keeps_run_fixed_when_runtime_is_not_ready() -> None:
    state = GuiScreenState(
        runtime_status=GuiRuntimeStatus(
            ready=False,
            headline="Setup",
            detail="missing",
            missing_items=("llama-server",),
        ),
        input_sup=Path("/tmp/sample.sup"),
        output_srt=Path("/tmp/sample.srt"),
    )

    action = derive_primary_action(state)

    assert action.kind == "convert"
    assert action.label == "Run"
    assert action.enabled is False


def test_derive_setup_action_keeps_neutral_setup_label_when_runtime_is_not_ready() -> None:
    state = GuiScreenState(
        runtime_status=GuiRuntimeStatus(
            ready=False,
            headline="Setup",
            detail="missing",
            missing_items=("llama-server",),
        ),
    )

    action = derive_setup_action(state)

    assert action.kind == "setup"
    assert action.label == "Set Up"
    assert action.enabled is True


def test_derive_setup_action_keeps_neutral_setup_label_when_runtime_is_ready() -> None:
    state = GuiScreenState(
        runtime_status=GuiRuntimeStatus(
            ready=True,
            headline="Ready",
            detail="go",
            missing_items=(),
        ),
    )

    action = derive_setup_action(state)

    assert action.kind == "setup"
    assert action.label == "Set Up"
    assert action.enabled is True


def test_derive_primary_action_requires_input_after_runtime_is_ready() -> None:
    state = GuiScreenState(
        runtime_status=GuiRuntimeStatus(
            ready=True,
            headline="Ready",
            detail="go",
            missing_items=(),
        ),
    )

    action = derive_primary_action(state)

    assert action.kind == "convert"
    assert action.label == "Run"
    assert action.enabled is False


def test_build_fast_convert_request_keeps_gui_convert_opinionated(tmp_path: Path) -> None:
    request = build_fast_convert_request(
        input_sup=tmp_path / "sample.sup",
        output_srt=tmp_path / "sample.srt",
        enable_furigana_mask=True,
        runtime_status=GuiRuntimeStatus(
            ready=True,
            headline="Ready",
            detail="",
            missing_items=(),
            runtime_binary_path=tmp_path / "llama-server.exe",
            models_dir=tmp_path / "models",
        ),
    )

    assert isinstance(request, ConvertRequest)
    assert request.engine == "llama-server"
    assert request.ocr_mode == "fast"
    assert request.corrector == "off"
    assert request.enable_furigana_mask is True
    assert request.runtime_binary_path == (tmp_path / "llama-server.exe")
    assert request.models_dir == (tmp_path / "models")


def test_build_setup_request_uses_default_runtime_assets_only() -> None:
    request = build_setup_request()

    assert isinstance(request, SetupRequest)
    assert request.with_hf_fallback is False
    assert request.with_qwen_corrector is False
    assert request.models_dir is not None
    assert request.derived_mmproj_output_path is not None
    assert request.bootstrap_managed_runtime is (os.name == "nt")
    assert request.install_prerequisites is False


def test_build_setup_request_for_variant_keeps_selected_runtime_variant() -> None:
    request = build_setup_request_for_variant(
        runtime_variant="x64/cpu",
        install_prerequisites=True,
    )

    assert isinstance(request, SetupRequest)
    assert request.runtime_variant == "x64/cpu"
    assert request.install_prerequisites is True


def test_format_runtime_facts_surfaces_target_and_managed_runtime_path(tmp_path: Path) -> None:
    facts = format_runtime_facts(
        status=GuiRuntimeStatus(
            ready=True,
            headline="Ready",
            detail="",
            missing_items=(),
            runtime_binary_path=tmp_path / "llama-server.exe",
            runtime_source="managed",
            runtime_release_tag="b8860",
            runtime_variant_id="x64/vulkan",
        ),
        selected_variant="auto",
    )

    assert "Target: auto" in facts
    assert "Managed: b8860 [x64/vulkan]" in facts
    assert "Path:" in facts


def test_probe_runtime_status_reports_missing_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import istots.gui.core as gui_core

    class _Assets:
        def __init__(self, model_path: Path, mmproj_path: Path) -> None:
            self.model_path = model_path
            self.mmproj_path = mmproj_path

    monkeypatch.setattr(
        gui_core,
        "resolve_gui_runtime_binding",
        lambda explicit_binary_path=None: type(
            "_Binding",
            (),
            {
                "source": "missing",
                "binary_path": None,
                "models_dir": tmp_path,
                "release_tag": None,
                "variant_id": None,
            },
        )(),
    )
    monkeypatch.setattr(gui_core, "describe_runtime_binding", lambda binding: "Managed runtime missing")
    monkeypatch.setattr(
        gui_core,
        "resolve_llama_server_role_assets",
        lambda role, **_: _Assets(
            model_path=tmp_path / f"{role}.gguf",
            mmproj_path=tmp_path / f"{role}.mmproj.gguf",
        ),
    )

    status = probe_runtime_status()

    assert status.ready is False
    assert "llama-server" in status.missing_items
    assert status.headline == "Setup"
    assert "Missing components:" in status.detail
    assert "- llama-server" in status.detail
    assert "Models:" in status.detail


def test_probe_runtime_status_marks_invalid_managed_runtime_as_not_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import istots.gui.core as gui_core

    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")

    class _Assets:
        def __init__(self, model_path: Path, mmproj_path: Path) -> None:
            self.model_path = model_path
            self.mmproj_path = mmproj_path

    monkeypatch.setattr(
        gui_core,
        "resolve_gui_runtime_binding",
        lambda explicit_binary_path=None: type(
            "_Binding",
            (),
            {
                "source": "managed",
                "binary_path": binary,
                "models_dir": tmp_path,
                "release_tag": "b8858",
                "variant_id": "x64/vulkan",
            },
        )(),
    )
    monkeypatch.setattr(gui_core, "describe_runtime_binding", lambda binding: "Managed runtime b8858 [x64/vulkan]")
    monkeypatch.setattr(
        gui_core,
        "resolve_llama_server_role_assets",
        lambda role, **_: _Assets(
            model_path=tmp_path / f"{role}.gguf",
            mmproj_path=tmp_path / f"{role}.mmproj.gguf",
        ),
    )
    for name in ("ocr.gguf", "ocr.mmproj.gguf", "ocr-fast.mmproj.gguf"):
        (tmp_path / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(
        gui_core,
        "validate_llama_server_binary",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "managed llama.cpp runtime failed startup validation.\n"
                f"Binary: {binary}\n"
                "Probe --version failed with exit=3221225477.\n"
                "Probe --help failed with exit=3221225477."
            )
        ),
    )

    status = probe_runtime_status()

    assert status.ready is False
    assert "runtime validation" in status.missing_items
    assert "Issues:" in status.detail
    assert "Runtime startup validation failed." in status.detail
    assert "Probe --version failed with exit=3221225477." in status.detail


def test_probe_runtime_status_does_not_treat_persisted_managed_validation_failure_as_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import istots.gui.core as gui_core

    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")

    class _Assets:
        def __init__(self, model_path: Path, mmproj_path: Path) -> None:
            self.model_path = model_path
            self.mmproj_path = mmproj_path

    monkeypatch.setattr(
        gui_core,
        "resolve_gui_runtime_binding",
        lambda explicit_binary_path=None: type(
            "_Binding",
            (),
            {
                "source": "managed",
                "binary_path": binary,
                "models_dir": tmp_path,
                "release_tag": "b8858",
                "variant_id": "x64/vulkan",
            },
        )(),
    )
    monkeypatch.setattr(gui_core, "describe_runtime_binding", lambda binding: "Managed runtime b8858 [x64/vulkan]")
    monkeypatch.setattr(
        gui_core,
        "resolve_llama_server_role_assets",
        lambda role, **_: _Assets(
            model_path=tmp_path / f"{role}.gguf",
            mmproj_path=tmp_path / f"{role}.mmproj.gguf",
        ),
    )
    for name in ("ocr.gguf", "ocr.mmproj.gguf", "ocr-fast.mmproj.gguf"):
        (tmp_path / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(gui_core, "validate_llama_server_binary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gui_core,
        "load_managed_runtime_state",
        lambda: type(
            "_State",
            (),
            {
                "binary_path": binary,
                "last_validation_ok": False,
                "last_validation_detail": "Runtime test needs attention.\n\nIssues:\n- ocr: llama-server exited before becoming ready",
            },
        )(),
    )

    status = probe_runtime_status()

    assert status.ready is True
    assert "runtime validation" not in status.missing_items
    assert "llama-server exited before becoming ready" not in status.detail


def test_probe_runtime_status_does_not_require_managed_runtime_startup_test_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import istots.gui.core as gui_core

    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")

    class _Assets:
        def __init__(self, model_path: Path, mmproj_path: Path) -> None:
            self.model_path = model_path
            self.mmproj_path = mmproj_path

    monkeypatch.setattr(
        gui_core,
        "resolve_gui_runtime_binding",
        lambda explicit_binary_path=None: type(
            "_Binding",
            (),
            {
                "source": "managed",
                "binary_path": binary,
                "models_dir": tmp_path,
                "release_tag": "b8860",
                "variant_id": "x64/vulkan",
            },
        )(),
    )
    monkeypatch.setattr(gui_core, "describe_runtime_binding", lambda binding: "Managed runtime b8860 [x64/vulkan]")
    monkeypatch.setattr(
        gui_core,
        "resolve_llama_server_role_assets",
        lambda role, **_: _Assets(
            model_path=tmp_path / f"{role}.gguf",
            mmproj_path=tmp_path / f"{role}.mmproj.gguf",
        ),
    )
    for name in ("ocr.gguf", "ocr.mmproj.gguf", "ocr-fast.mmproj.gguf"):
        (tmp_path / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(gui_core, "validate_llama_server_binary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gui_core,
        "load_managed_runtime_state",
        lambda: type(
            "_State",
            (),
            {
                "binary_path": binary,
                "last_validation_ok": None,
                "last_validation_detail": "",
            },
        )(),
    )

    status = probe_runtime_status()

    assert status.ready is True
    assert "runtime validation" not in status.missing_items
    assert "Managed runtime has not completed a startup test on this machine yet." not in status.detail


def test_run_gui_doctor_check_passes_only_when_ocr_and_fast_smoke_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    import istots.gui.core as gui_core

    class _Report:
        def __init__(self, role: str) -> None:
            self.role = role
            self.launch_spec = None
            self.issues = ()
            self.ok = True

    seen_roles: list[str] = []
    seen_ctx_sizes: list[int | None] = []

    def _fake_doctor(*, role: str, **kwargs):
        seen_roles.append(role)
        seen_ctx_sizes.append(kwargs["overrides"].ctx_size)
        return _Report(role)

    monkeypatch.setattr(
        gui_core,
        "resolve_gui_runtime_binding",
        lambda explicit_binary_path=None: type(
            "_Binding",
            (),
            {
                "source": "managed",
                "binary_path": Path("/tmp/llama-server"),
                "models_dir": Path("/tmp/models"),
                "release_tag": "b8855",
                "variant_id": "x64/cpu",
            },
        )(),
    )
    monkeypatch.setattr(gui_core, "run_llama_server_doctor", _fake_doctor)

    status = run_gui_doctor_check()

    assert seen_roles == ["ocr", "ocr-fast"]
    assert seen_ctx_sizes == [gui_core.LOCAL_PADDLE_CTX_SIZE, gui_core.LOCAL_PADDLE_CTX_SIZE]
    assert status.ready is True
    assert "Runtime test passed." in status.detail
    assert "Runtime:" in status.detail


def test_run_gui_doctor_check_surfaces_role_specific_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import istots.gui.core as gui_core

    class _Issue:
        def __init__(self, code: str, message: str) -> None:
            self.code = code
            self.message = message

    class _Report:
        def __init__(self, role: str, issues: tuple[_Issue, ...]) -> None:
            self.role = role
            self.launch_spec = None
            self.issues = issues
            self.ok = not issues

    def _fake_doctor(*, role: str, **kwargs):
        if role == "ocr-fast":
            return _Report(role, (_Issue("port_in_use", "requested port is already in use: 127.0.0.1:8081"),))
        return _Report(role, ())

    monkeypatch.setattr(
        gui_core,
        "resolve_gui_runtime_binding",
        lambda explicit_binary_path=None: type(
            "_Binding",
            (),
            {
                "source": "managed",
                "binary_path": Path("/tmp/llama-server"),
                "models_dir": Path("/tmp/models"),
                "release_tag": "b8855",
                "variant_id": "x64/cpu",
            },
        )(),
    )
    monkeypatch.setattr(gui_core, "run_llama_server_doctor", _fake_doctor)

    status = run_gui_doctor_check()

    assert status.ready is False
    assert status.headline == "Check"
    assert "Runtime test needs attention." in status.detail
    assert "- ocr-fast: requested port is already in use" in status.detail
    assert "Issues:" in status.detail
    assert "ocr-fast:port_in_use" in status.missing_items


def test_run_gui_doctor_check_passes_cancel_event_to_runtime_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    import istots.gui.core as gui_core

    class _Report:
        def __init__(self) -> None:
            self.role = "ocr"
            self.launch_spec = None
            self.issues = ()
            self.ok = True

    seen_cancel_events: list[object] = []

    monkeypatch.setattr(
        gui_core,
        "resolve_gui_runtime_binding",
        lambda explicit_binary_path=None: type(
            "_Binding",
            (),
            {
                "source": "managed",
                "binary_path": Path("/tmp/llama-server"),
                "models_dir": Path("/tmp/models"),
                "release_tag": "b8855",
                "variant_id": "x64/cpu",
            },
        )(),
    )
    monkeypatch.setattr(
        gui_core,
        "run_llama_server_doctor",
        lambda **kwargs: seen_cancel_events.append(kwargs["cancel_event"]) or _Report(),
    )

    cancel_event = threading.Event()
    status = run_gui_doctor_check(cancel_event=cancel_event)

    assert status.ready is True
    assert seen_cancel_events == [cancel_event, cancel_event]


def test_format_setup_summary_keeps_setup_lane_single_line_and_targeted(tmp_path: Path) -> None:
    summary = format_setup_summary(
        status=GuiRuntimeStatus(
            ready=True,
            headline="Ready",
            detail="",
            missing_items=(),
            runtime_binary_path=tmp_path / "llama-server.exe",
            runtime_source="managed",
            runtime_release_tag="b8860",
            runtime_variant_id="x64/vulkan",
        ),
        selected_variant="auto",
    )

    assert summary == "Done | Managed b8860 [x64/vulkan] | target auto"


def test_format_check_summary_compacts_multiline_failure_detail() -> None:
    summary = format_check_summary(
        state="fail",
        detail=(
            "Runtime test needs attention.\n\n"
            "Runtime:\n"
            "- Managed runtime\n"
            "Issues:\n"
            "- ocr-fast: startup_failed"
        ),
    )

    assert summary == "Failed | ocr-fast: startup_failed"
