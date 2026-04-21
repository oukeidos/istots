from __future__ import annotations

import os

from istots import frozen_subprocess


def test_sanitized_external_subprocess_env_is_noop_when_not_frozen(monkeypatch) -> None:
    monkeypatch.delattr(frozen_subprocess.sys, "frozen", raising=False)

    assert frozen_subprocess.sanitized_external_subprocess_env() is None
    assert frozen_subprocess.sanitized_external_subprocess_env({"A": "1"}) == {"A": "1"}


def test_sanitized_external_subprocess_env_restores_original_linux_library_path(
    monkeypatch,
    tmp_path,
) -> None:
    app_root = (tmp_path / "bundle").resolve()
    app_root.mkdir()
    monkeypatch.setattr(frozen_subprocess.sys, "platform", "linux")
    monkeypatch.setattr(frozen_subprocess.sys, "frozen", True, raising=False)
    monkeypatch.setattr(frozen_subprocess.sys, "_MEIPASS", str(app_root), raising=False)

    env = frozen_subprocess.sanitized_external_subprocess_env(
        {
            "LD_LIBRARY_PATH": f"{app_root}{os.pathsep}/usr/lib",
            "LD_LIBRARY_PATH_ORIG": "/usr/lib",
        }
    )

    assert env is not None
    assert env["LD_LIBRARY_PATH"] == "/usr/lib"


def test_sanitized_external_subprocess_runtime_clears_windows_dll_directory(
    monkeypatch,
    tmp_path,
) -> None:
    app_root = (tmp_path / "bundle").resolve()
    qt_dir = (app_root / "qt").resolve()
    app_root.mkdir()
    qt_dir.mkdir()

    monkeypatch.setattr(frozen_subprocess.sys, "platform", "win32")
    monkeypatch.setattr(frozen_subprocess.sys, "frozen", True, raising=False)
    monkeypatch.setattr(frozen_subprocess.sys, "_MEIPASS", str(app_root), raising=False)

    seen_calls: list[str | None] = []
    monkeypatch.setattr(
        frozen_subprocess,
        "_set_windows_dll_directory",
        lambda path: seen_calls.append(path),
    )

    with frozen_subprocess.sanitized_external_subprocess_runtime(
        {"PATH": os.pathsep.join((str(qt_dir), r"C:\Windows\System32"))}
    ) as env:
        assert env is not None
        assert env["PATH"] == r"C:\Windows\System32"

    assert seen_calls == [None, str(app_root)]
