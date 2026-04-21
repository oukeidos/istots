from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from istots import runtime_prerequisites


def test_probe_windows_vc_redist_x64_prefers_registry_version(monkeypatch) -> None:
    monkeypatch.setattr(runtime_prerequisites, "_read_windows_vc_redist_registry_version", lambda arch: "14.44.35211.0")

    status = runtime_prerequisites._probe_windows_vc_redist_x64()

    assert status.ok is True
    assert status.installed_version == "14.44.35211.0"


def test_probe_windows_vc_redist_x64_requires_registry_installation(monkeypatch) -> None:
    monkeypatch.setattr(runtime_prerequisites, "_read_windows_vc_redist_registry_version", lambda arch: None)

    status = runtime_prerequisites._probe_windows_vc_redist_x64()

    assert status.ok is False
    assert "was not detected" in status.detail


def test_probe_windows_vc_redist_x64_reports_missing_runtime(monkeypatch) -> None:
    monkeypatch.setattr(runtime_prerequisites, "_read_windows_vc_redist_registry_version", lambda arch: None)

    status = runtime_prerequisites._probe_windows_vc_redist_x64()

    assert status.ok is False
    assert status.installer_url == runtime_prerequisites.WINDOWS_VC_REDIST_X64_URL


def test_ensure_managed_runtime_prerequisites_raises_when_install_disabled(monkeypatch, tmp_path: Path) -> None:
    missing = runtime_prerequisites.RuntimePrerequisiteStatus(
        key="windows-msvc-v14-x64",
        label="Microsoft Visual C++ Redistributable (x64)",
        ok=False,
        detail="missing",
        remediation="install it",
        installer_url="https://example.invalid/vc_redist.x64.exe",
    )

    monkeypatch.setattr(runtime_prerequisites, "missing_managed_runtime_prerequisites", lambda: (missing,))

    with pytest.raises(RuntimeError, match="Managed runtime prerequisite check failed."):
        runtime_prerequisites.ensure_managed_runtime_prerequisites(
            install=False,
            download_root=tmp_path,
        )


def test_ensure_managed_runtime_prerequisites_installs_missing_items(monkeypatch, tmp_path: Path) -> None:
    missing = runtime_prerequisites.RuntimePrerequisiteStatus(
        key="windows-msvc-v14-x64",
        label="Microsoft Visual C++ Redistributable (x64)",
        ok=False,
        detail="missing",
        remediation="install it",
        installer_url="https://example.invalid/vc_redist.x64.exe",
    )
    seen_installs: list[Path] = []
    probe_results = [(missing,), ()]

    monkeypatch.setattr(runtime_prerequisites, "missing_managed_runtime_prerequisites", lambda: probe_results.pop(0))
    monkeypatch.setattr(
        runtime_prerequisites,
        "_install_windows_vc_redist_x64",
        lambda *, download_root, progress_callback, cancel_event=None: seen_installs.append(download_root),
    )

    runtime_prerequisites.ensure_managed_runtime_prerequisites(
        install=True,
        download_root=tmp_path / "downloads",
    )

    assert seen_installs == [(tmp_path / "downloads").resolve()]


def test_download_file_reuses_existing_installer(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "vc_redist.x64.exe"
    target.write_bytes(b"existing-installer")

    def _unexpected_urlopen(*args, **kwargs):
        raise AssertionError("download should have been skipped")

    monkeypatch.setattr(runtime_prerequisites.urllib.request, "urlopen", _unexpected_urlopen)

    runtime_prerequisites._download_file(
        "https://example.invalid/vc_redist.x64.exe",
        target,
        progress_callback=None,
    )


def test_install_windows_vc_redist_uses_sanitized_subprocess_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "vc_redist.x64.exe"
    installer.write_bytes(b"binary")
    seen_kwargs: list[dict[str, object]] = []

    @contextmanager
    def _fake_subprocess_runtime():
        yield {"PATH": "clean"}

    def _fake_download(url: str, target_path: Path, *, progress_callback, cancel_event=None) -> None:
        target_path.write_bytes(installer.read_bytes())

    monkeypatch.setattr(runtime_prerequisites, "_download_file", _fake_download)
    monkeypatch.setattr(
        runtime_prerequisites,
        "sanitized_external_subprocess_runtime",
        _fake_subprocess_runtime,
    )
    monkeypatch.setattr(
        runtime_prerequisites.subprocess,
        "Popen",
        lambda *args, **kwargs: seen_kwargs.append(kwargs) or SimpleNamespace(args=args[0]),
    )
    monkeypatch.setattr(
        runtime_prerequisites,
        "_wait_for_prerequisite_installer",
        lambda process, *, cancel_event: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    runtime_prerequisites._install_windows_vc_redist_x64(
        download_root=tmp_path,
        progress_callback=None,
    )

    assert seen_kwargs[0]["env"] == {"PATH": "clean"}
