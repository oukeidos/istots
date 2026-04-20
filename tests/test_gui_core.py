from __future__ import annotations

from pathlib import Path

import pytest

from istots.app.convert import ConvertRequest
from istots.app.setup import SetupRequest
from istots.gui.core import (
    GuiRuntimeStatus,
    GuiScreenState,
    build_fast_convert_request,
    build_setup_request,
    derive_primary_action,
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


def test_derive_primary_action_prefers_setup_when_runtime_is_not_ready() -> None:
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

    assert action.kind == "setup"
    assert action.label == "Setup"
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
    )

    assert isinstance(request, ConvertRequest)
    assert request.engine == "llama-server"
    assert request.ocr_mode == "fast"
    assert request.corrector == "off"
    assert request.enable_furigana_mask is True


def test_build_setup_request_uses_default_runtime_assets_only() -> None:
    request = build_setup_request()

    assert isinstance(request, SetupRequest)
    assert request.with_hf_fallback is False
    assert request.with_qwen_corrector is False


def test_probe_runtime_status_reports_missing_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import istots.gui.core as gui_core

    class _Assets:
        def __init__(self, model_path: Path, mmproj_path: Path) -> None:
            self.model_path = model_path
            self.mmproj_path = mmproj_path

    monkeypatch.setattr(gui_core, "detect_llama_server_path", lambda _: None)
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

    monkeypatch.setattr(gui_core, "run_llama_server_doctor", _fake_doctor)

    status = run_gui_doctor_check()

    assert seen_roles == ["ocr", "ocr-fast"]
    assert seen_ctx_sizes == [gui_core.LOCAL_PADDLE_CTX_SIZE, gui_core.LOCAL_PADDLE_CTX_SIZE]
    assert status.ready is True
    assert status.detail == "OK"


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

    monkeypatch.setattr(gui_core, "run_llama_server_doctor", _fake_doctor)

    status = run_gui_doctor_check()

    assert status.ready is False
    assert status.headline == "Check"
    assert "ocr-fast: requested port is already in use" in status.detail
    assert "ocr-fast:port_in_use" in status.missing_items
