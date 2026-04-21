from __future__ import annotations

import json
from pathlib import Path

from istots import runtime_diagnostics


def test_append_runtime_diagnostic_event_writes_jsonl(monkeypatch, tmp_path: Path) -> None:
    managed_root = tmp_path / "managed-root"
    monkeypatch.setenv(runtime_diagnostics.GUI_MANAGED_ROOT_ENV, str(managed_root))

    log_path = runtime_diagnostics.append_runtime_diagnostic_event(
        "setup_assets_begin",
        models_dir=tmp_path / "models",
        values={"count": 2},
    )

    assert log_path == managed_root / "state" / runtime_diagnostics.RUNTIME_DIAGNOSTICS_FILENAME
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "setup_assets_begin"
    assert payload["models_dir"] == str((tmp_path / "models"))
    assert payload["values"] == {"count": 2}
    assert payload["pid"] > 0


def test_install_faulthandler_trace_uses_managed_state_dir(monkeypatch, tmp_path: Path) -> None:
    managed_root = tmp_path / "managed-root"
    monkeypatch.setenv(runtime_diagnostics.GUI_MANAGED_ROOT_ENV, str(managed_root))
    runtime_diagnostics._close_faulthandler_stream()

    path = runtime_diagnostics.install_faulthandler_trace()

    assert path == managed_root / "state" / runtime_diagnostics.PYTHON_FAULTHANDLER_FILENAME
    assert path.exists()
    runtime_diagnostics._close_faulthandler_stream()
