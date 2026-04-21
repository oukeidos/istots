from __future__ import annotations

import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows only
    winreg = None


WINDOWS_VC_REDIST_X64_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


@dataclass(frozen=True)
class RuntimePrerequisiteStatus:
    key: str
    label: str
    ok: bool
    detail: str
    remediation: str = ""
    installer_url: str | None = None
    installed_version: str | None = None
    restart_required: bool = False


def probe_managed_runtime_prerequisites() -> tuple[RuntimePrerequisiteStatus, ...]:
    if os.name == "nt":
        return (_probe_windows_vc_redist_x64(),)
    return ()


def missing_managed_runtime_prerequisites() -> tuple[RuntimePrerequisiteStatus, ...]:
    return tuple(item for item in probe_managed_runtime_prerequisites() if not item.ok)


def ensure_managed_runtime_prerequisites(
    *,
    install: bool,
    download_root: Path,
    progress_callback: Callable[[str, str, str, float | None], None] | None = None,
) -> tuple[RuntimePrerequisiteStatus, ...]:
    missing = missing_managed_runtime_prerequisites()
    if not missing:
        _report_progress(
            progress_callback,
            phase="prereq_validate",
            headline="Check Prerequisites",
            detail="Runtime prerequisites already satisfied",
            fraction=0.04,
        )
        return ()
    if not install:
        raise RuntimeError(format_missing_managed_runtime_prerequisites(missing))

    _report_progress(
        progress_callback,
        phase="prereq_validate",
        headline="Check Prerequisites",
        detail="Installing runtime prerequisites",
        fraction=0.02,
    )
    download_root = download_root.expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    for item in missing:
        if item.key == "windows-msvc-v14-x64":
            _install_windows_vc_redist_x64(
                download_root=download_root,
                progress_callback=progress_callback,
            )
            continue
        raise RuntimeError(format_missing_managed_runtime_prerequisites((item,)))

    remaining = missing_managed_runtime_prerequisites()
    if remaining:
        raise RuntimeError(format_missing_managed_runtime_prerequisites(remaining))

    _report_progress(
        progress_callback,
        phase="prereq_validate",
        headline="Check Prerequisites",
        detail="Runtime prerequisites are ready",
        fraction=0.10,
    )
    return ()


def format_missing_managed_runtime_prerequisites(
    items: tuple[RuntimePrerequisiteStatus, ...],
) -> str:
    lines = ["Managed runtime prerequisite check failed."]
    for item in items:
        lines.append(f"{item.label}: {item.detail}")
        if item.installed_version:
            lines.append(f"Detected version: {item.installed_version}")
        if item.remediation:
            lines.append(f"Remediation: {item.remediation}")
        if item.installer_url:
            lines.append(f"Download: {item.installer_url}")
    return "\n".join(lines)


def _probe_windows_vc_redist_x64() -> RuntimePrerequisiteStatus:
    installed_version = _read_windows_vc_redist_registry_version("x64")
    if installed_version is not None:
        return RuntimePrerequisiteStatus(
            key="windows-msvc-v14-x64",
            label="Microsoft Visual C++ Redistributable (x64)",
            ok=True,
            detail="Installed",
            installed_version=installed_version,
        )
    return RuntimePrerequisiteStatus(
        key="windows-msvc-v14-x64",
        label="Microsoft Visual C++ Redistributable (x64)",
        ok=False,
        detail="The MSVC v14 runtime required by the official Windows llama.cpp build was not detected.",
        remediation="Install or repair the latest supported Microsoft Visual C++ Redistributable (x64), then run Setup again.",
        installer_url=WINDOWS_VC_REDIST_X64_URL,
    )


def _install_windows_vc_redist_x64(
    *,
    download_root: Path,
    progress_callback: Callable[[str, str, str, float | None], None] | None,
) -> None:
    installer_path = (download_root / "vc_redist.x64.exe").resolve()
    log_path = (download_root / "vc_redist.x64.log").resolve()

    _report_progress(
        progress_callback,
        phase="prereq_download",
        headline="Download Prerequisite",
        detail="Microsoft Visual C++ Redistributable (x64)",
        fraction=0.04,
    )
    _download_file(WINDOWS_VC_REDIST_X64_URL, installer_path, progress_callback=progress_callback)

    _report_progress(
        progress_callback,
        phase="prereq_install",
        headline="Install Prerequisite",
        detail="Microsoft Visual C++ Redistributable (x64)",
        fraction=0.08,
    )
    completed = subprocess.run(
        [
            str(installer_path),
            "/install",
            "/passive",
            "/norestart",
            "/log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode not in {0, 3010, 1638}:
        raise RuntimeError(
            "Microsoft Visual C++ Redistributable (x64) installation failed.\n"
            f"Installer exit code: {completed.returncode}\n"
            f"Log: {log_path}\n"
            f"stdout: {(completed.stdout or '').strip()[-400:]}\n"
            f"stderr: {(completed.stderr or '').strip()[-400:]}"
        )


def _download_file(
    url: str,
    target_path: Path,
    *,
    progress_callback: Callable[[str, str, str, float | None], None] | None,
) -> None:
    if target_path.exists() and target_path.stat().st_size > 0:
        _report_progress(
            progress_callback,
            phase="prereq_download",
            headline="Download Prerequisite",
            detail=f"Reusing {target_path.name}",
            fraction=0.08,
        )
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "istots-gui-bootstrap"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        total_bytes = int(response.headers.get("Content-Length") or 0)
        downloaded_bytes = 0
        with target_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 128)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded_bytes += len(chunk)
                fraction = None
                detail = target_path.name
                if total_bytes > 0:
                    fraction = 0.04 + (min(downloaded_bytes / total_bytes, 1.0) * 0.04)
                    detail = (
                        f"{target_path.name} "
                        f"{downloaded_bytes / (1024 * 1024):.1f}/{total_bytes / (1024 * 1024):.1f} MB"
                    )
                _report_progress(
                    progress_callback,
                    phase="prereq_download",
                    headline="Download Prerequisite",
                    detail=detail,
                    fraction=fraction,
                )


def _read_windows_vc_redist_registry_version(arch: str) -> str | None:
    if os.name != "nt" or winreg is None:
        return None
    registry_paths = (
        fr"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\{arch}",
        fr"SOFTWARE\Wow6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\{arch}",
    )
    registry_views = (
        getattr(winreg, "KEY_WOW64_64KEY", 0),
        getattr(winreg, "KEY_WOW64_32KEY", 0),
        0,
    )
    for registry_path in registry_paths:
        for registry_view in registry_views:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    registry_path,
                    0,
                    winreg.KEY_READ | registry_view,
                ) as key:
                    installed = int(winreg.QueryValueEx(key, "Installed")[0])
                    if installed != 1:
                        continue
                    version = str(winreg.QueryValueEx(key, "Version")[0]).strip()
                    if version:
                        return version
                    return "installed"
            except OSError:
                continue
    return None


def _report_progress(
    callback: Callable[[str, str, str, float | None], None] | None,
    *,
    phase: str,
    headline: str,
    detail: str,
    fraction: float | None,
) -> None:
    if callback is None:
        return
    callback(phase, headline, detail, fraction)
