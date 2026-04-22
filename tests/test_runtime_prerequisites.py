from __future__ import annotations

import threading
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


def test_download_file_redownloads_existing_installer(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "vc_redist.x64.exe"
    target.write_bytes(b"existing-installer")
    seen_urls: list[str] = []

    class _Response:
        headers = {"Content-Length": "15"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_done"):
                return b""
            self._done = True
            return b"fresh-installer"

    monkeypatch.setattr(
        runtime_prerequisites.urllib.request,
        "urlopen",
        lambda request, timeout=300: seen_urls.append(request.full_url) or _Response(),
    )

    runtime_prerequisites._download_file(
        "https://example.invalid/vc_redist.x64.exe",
        target,
        progress_callback=None,
    )

    assert seen_urls == ["https://example.invalid/vc_redist.x64.exe"]
    assert target.read_bytes() == b"fresh-installer"
    assert not (tmp_path / "vc_redist.x64.exe.part").exists()


def test_download_file_discards_stale_partial_before_retry(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "vc_redist.x64.exe"
    temp_target = tmp_path / "vc_redist.x64.exe.part"
    temp_target.write_bytes(b"stale-partial-installer")

    class _Response:
        headers = {"Content-Length": "13"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_done"):
                return b""
            self._done = True
            return b"fresh-package"

    monkeypatch.setattr(
        runtime_prerequisites.urllib.request,
        "urlopen",
        lambda request, timeout=300: _Response(),
    )

    runtime_prerequisites._download_file(
        "https://example.invalid/vc_redist.x64.exe",
        target,
        progress_callback=None,
    )

    assert target.read_bytes() == b"fresh-package"
    assert not temp_target.exists()


def test_download_file_cleans_partial_files_after_cancellation(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "vc_redist.x64.exe"
    temp_target = tmp_path / "vc_redist.x64.exe.part"
    cancel_event = threading.Event()

    class _Response:
        headers = {"Content-Length": "14"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_done"):
                return b""
            self._done = True
            cancel_event.set()
            return b"partial"

    monkeypatch.setattr(
        runtime_prerequisites.urllib.request,
        "urlopen",
        lambda request, timeout=300: _Response(),
    )

    with pytest.raises(RuntimeError, match="cancelled during prerequisite download"):
        runtime_prerequisites._download_file(
            "https://example.invalid/vc_redist.x64.exe",
            target,
            progress_callback=None,
            cancel_event=cancel_event,
        )

    assert not target.exists()
    assert not temp_target.exists()


def test_download_file_rejects_incomplete_response(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "vc_redist.x64.exe"
    temp_target = tmp_path / "vc_redist.x64.exe.part"

    class _Response:
        headers = {"Content-Length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_done"):
                return b""
            self._done = True
            return b"short"

    monkeypatch.setattr(
        runtime_prerequisites.urllib.request,
        "urlopen",
        lambda request, timeout=300: _Response(),
    )

    with pytest.raises(RuntimeError, match="download incomplete"):
        runtime_prerequisites._download_file(
            "https://example.invalid/vc_redist.x64.exe",
            target,
            progress_callback=None,
        )

    assert not target.exists()
    assert not temp_target.exists()


def test_verify_windows_vc_redist_signature_accepts_valid_microsoft_signature(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "vc_redist.x64.exe"
    installer.write_bytes(b"binary")

    monkeypatch.setattr(
        runtime_prerequisites,
        "_read_windows_authenticode_signature_info",
        lambda installer_path, *, cancel_event=None: {
            "status": "Valid",
            "status_message": "",
            "subject": "CN=Microsoft Corporation, O=Microsoft Corporation",
        },
    )

    runtime_prerequisites._verify_windows_vc_redist_signature(installer)


def test_verify_windows_vc_redist_signature_rejects_invalid_status(monkeypatch, tmp_path: Path) -> None:
    installer = tmp_path / "vc_redist.x64.exe"
    installer.write_bytes(b"binary")

    monkeypatch.setattr(
        runtime_prerequisites,
        "_read_windows_authenticode_signature_info",
        lambda installer_path, *, cancel_event=None: {
            "status": "HashMismatch",
            "status_message": "The file hash does not match.",
            "subject": "CN=Microsoft Corporation, O=Microsoft Corporation",
        },
    )

    with pytest.raises(RuntimeError, match="signature verification failed"):
        runtime_prerequisites._verify_windows_vc_redist_signature(installer)


def test_verify_windows_vc_redist_signature_rejects_non_microsoft_signer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "vc_redist.x64.exe"
    installer.write_bytes(b"binary")

    monkeypatch.setattr(
        runtime_prerequisites,
        "_read_windows_authenticode_signature_info",
        lambda installer_path, *, cancel_event=None: {
            "status": "Valid",
            "status_message": "",
            "subject": "CN=Contoso Test Signing",
        },
    )

    with pytest.raises(RuntimeError, match="expected a Microsoft signer"):
        runtime_prerequisites._verify_windows_vc_redist_signature(installer)


def test_read_windows_authenticode_signature_info_uses_sanitized_subprocess_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "vc_redist.x64.exe"
    installer.write_bytes(b"binary")
    seen_kwargs: list[dict[str, object]] = []

    @contextmanager
    def _fake_subprocess_runtime():
        yield {"PATH": "clean"}

    monkeypatch.setattr(
        runtime_prerequisites,
        "sanitized_external_subprocess_runtime",
        _fake_subprocess_runtime,
    )
    monkeypatch.setattr(runtime_prerequisites.os, "name", "nt")
    monkeypatch.setattr(
        runtime_prerequisites.subprocess,
        "Popen",
        lambda *args, **kwargs: seen_kwargs.append(kwargs) or SimpleNamespace(args=args[0]),
    )
    monkeypatch.setattr(
        runtime_prerequisites,
        "_wait_for_prerequisite_signature_check",
        lambda process, *, cancel_event: SimpleNamespace(
            returncode=0,
            stdout='{"Status":"Valid","StatusMessage":"","Subject":"CN=Microsoft Corporation"}',
            stderr="",
        ),
    )

    payload = runtime_prerequisites._read_windows_authenticode_signature_info(installer)

    assert payload == {
        "status": "Valid",
        "status_message": "",
        "subject": "CN=Microsoft Corporation",
    }
    assert seen_kwargs[0]["env"] == {"PATH": "clean"}
    assert seen_kwargs[0]["creationflags"] & getattr(runtime_prerequisites.subprocess, "CREATE_NO_WINDOW", 0)


def test_read_windows_authenticode_signature_info_rejects_failed_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "vc_redist.x64.exe"
    installer.write_bytes(b"binary")

    @contextmanager
    def _fake_subprocess_runtime():
        yield {"PATH": "clean"}

    monkeypatch.setattr(
        runtime_prerequisites,
        "sanitized_external_subprocess_runtime",
        _fake_subprocess_runtime,
    )
    monkeypatch.setattr(runtime_prerequisites.os, "name", "nt")
    monkeypatch.setattr(
        runtime_prerequisites.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(args=args[0]),
    )
    monkeypatch.setattr(
        runtime_prerequisites,
        "_wait_for_prerequisite_signature_check",
        lambda process, *, cancel_event: SimpleNamespace(returncode=1, stdout="", stderr="powershell failed"),
    )

    with pytest.raises(RuntimeError, match="signature verification failed"):
        runtime_prerequisites._read_windows_authenticode_signature_info(installer)


def test_install_windows_vc_redist_uses_sanitized_subprocess_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installer = tmp_path / "vc_redist.x64.exe"
    installer.write_bytes(b"binary")
    seen_kwargs: list[dict[str, object]] = []
    seen_signature_checks: list[Path] = []

    @contextmanager
    def _fake_subprocess_runtime():
        yield {"PATH": "clean"}

    def _fake_download(url: str, target_path: Path, *, progress_callback, cancel_event=None) -> None:
        target_path.write_bytes(installer.read_bytes())

    monkeypatch.setattr(runtime_prerequisites, "_download_file", _fake_download)
    monkeypatch.setattr(
        runtime_prerequisites,
        "_verify_windows_vc_redist_signature",
        lambda installer_path, *, cancel_event=None: seen_signature_checks.append(installer_path),
    )
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

    assert seen_signature_checks == [(tmp_path / "vc_redist.x64.exe").resolve()]
    assert seen_kwargs[0]["env"] == {"PATH": "clean"}
    assert seen_kwargs[0]["creationflags"] & getattr(runtime_prerequisites.subprocess, "CREATE_NO_WINDOW", 0)
