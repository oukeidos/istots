from __future__ import annotations

from pathlib import Path

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
        lambda *, download_root, progress_callback: seen_installs.append(download_root),
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
