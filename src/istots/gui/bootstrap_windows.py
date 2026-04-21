from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from istots.llama_mmproj import default_materialized_mmproj_path
from istots.runtime_prerequisites import (
    ensure_managed_runtime_prerequisites,
    format_missing_managed_runtime_prerequisites,
    missing_managed_runtime_prerequisites,
)

LLAMA_CPP_LATEST_RELEASE_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
GUI_MANAGED_ROOT_ENV = "ISTOTS_GUI_MANAGED_ROOT"
GUI_RUNTIME_STATE_FILENAME = "llama_cpp_runtime.json"
MANAGED_RUNTIME_SOURCE = "managed"
EXTERNAL_RUNTIME_SOURCE = "external"
OVERRIDE_RUNTIME_SOURCE = "override"
MISSING_RUNTIME_SOURCE = "missing"
MANUAL_RUNTIME_VARIANTS = (
    "x64/cpu",
    "arm64/cpu",
    "x64/cuda12",
    "x64/cuda13",
    "x64/vulkan",
    "x64/sycl",
    "x64/hip",
)
AUTO_RUNTIME_VARIANT_FALLBACK = "x64/cpu"

_VARIANT_PRIMARY_PATTERNS: dict[str, re.Pattern[str]] = {
    "x64/cpu": re.compile(r"^llama-.*-bin-win-cpu-x64\.zip$", re.IGNORECASE),
    "arm64/cpu": re.compile(r"^llama-.*-bin-win-cpu-arm64\.zip$", re.IGNORECASE),
    "x64/cuda12": re.compile(r"^llama-.*-bin-win-cuda-12\.\d+-x64\.zip$", re.IGNORECASE),
    "x64/cuda13": re.compile(r"^llama-.*-bin-win-cuda-13\.\d+-x64\.zip$", re.IGNORECASE),
    "x64/vulkan": re.compile(r"^llama-.*-bin-win-vulkan-x64\.zip$", re.IGNORECASE),
    "x64/sycl": re.compile(r"^llama-.*-bin-win-sycl-x64\.zip$", re.IGNORECASE),
    "x64/hip": re.compile(r"^llama-.*-bin-win-hip-x64\.zip$", re.IGNORECASE),
}
_VARIANT_COMPANION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "x64/cuda12": (
        re.compile(r"^llama-.*-cuda-12\.\d+-dlls-x64\.zip$", re.IGNORECASE),
    ),
    "x64/cuda13": (
        re.compile(r"^llama-.*-cuda-13\.\d+-dlls-x64\.zip$", re.IGNORECASE),
    ),
}


@dataclass(frozen=True)
class GuiManagedPaths:
    root: Path
    models_dir: Path
    runtime_dir: Path
    derived_mmproj_dir: Path
    state_dir: Path
    runtime_state_path: Path


@dataclass(frozen=True)
class LlamaCppReleaseAsset:
    name: str
    download_url: str
    size_bytes: int


@dataclass(frozen=True)
class LlamaCppReleaseCatalog:
    tag_name: str
    assets: tuple[LlamaCppReleaseAsset, ...]


@dataclass(frozen=True)
class ManagedLlamaCppRuntimeState:
    release_tag: str
    variant_id: str
    install_dir: Path
    binary_path: Path
    preferred_source: str = MANAGED_RUNTIME_SOURCE
    installed_at: float = 0.0
    last_validation_ok: bool | None = None
    last_validation_detail: str = ""
    last_validated_at: float = 0.0


@dataclass(frozen=True)
class GuiRuntimeBinding:
    source: str
    binary_path: Path | None
    models_dir: Path
    release_tag: str | None = None
    variant_id: str | None = None
    install_dir: Path | None = None


def gui_managed_paths() -> GuiManagedPaths:
    configured_root = os.environ.get(GUI_MANAGED_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            root = Path(local_app_data).expanduser().resolve() / "istots" / "managed"
        else:
            root = (Path.home() / "AppData" / "Local" / "istots" / "managed").resolve()
    elif sys_platform() == "darwin":
        root = (Path.home() / "Library" / "Application Support" / "istots" / "managed").resolve()
    else:
        root = (Path.home() / ".local" / "share" / "istots" / "managed").resolve()

    state_dir = root / "state"
    return GuiManagedPaths(
        root=root,
        models_dir=root / "models",
        runtime_dir=root / "runtime" / "llama.cpp",
        derived_mmproj_dir=root / "derived" / "mmproj",
        state_dir=state_dir,
        runtime_state_path=state_dir / GUI_RUNTIME_STATE_FILENAME,
    )


def gui_managed_root() -> Path:
    return gui_managed_paths().root


def gui_managed_models_dir() -> Path:
    return gui_managed_paths().models_dir


def gui_managed_runtime_dir() -> Path:
    return gui_managed_paths().runtime_dir


def gui_managed_derived_mmproj_dir() -> Path:
    return gui_managed_paths().derived_mmproj_dir


def use_managed_fast_mmproj_for_models_dir(models_dir: Path | None) -> bool:
    if models_dir is None:
        return False
    return models_dir.expanduser().resolve() == gui_managed_models_dir().resolve()


def managed_fast_mmproj_path(
    base_mmproj: Path,
    *,
    models_dir: Path | None,
    min_pixels: int,
) -> Path | None:
    if not use_managed_fast_mmproj_for_models_dir(models_dir):
        return None
    filename = default_materialized_mmproj_path(base_mmproj, min_pixels).name
    return (gui_managed_derived_mmproj_dir() / base_mmproj.parent.name / filename).resolve()


def load_managed_runtime_state() -> ManagedLlamaCppRuntimeState | None:
    path = gui_managed_paths().runtime_state_path
    try:
        if not path.exists():
            return None
    except OSError:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ManagedLlamaCppRuntimeState(
            release_tag=str(payload["release_tag"]),
            variant_id=str(payload["variant_id"]),
            install_dir=Path(str(payload["install_dir"])).expanduser().resolve(),
            binary_path=Path(str(payload["binary_path"])).expanduser().resolve(),
            preferred_source=str(payload.get("preferred_source", MANAGED_RUNTIME_SOURCE)),
            installed_at=float(payload.get("installed_at", 0.0)),
            last_validation_ok=(
                None
                if payload.get("last_validation_ok") is None
                else bool(payload.get("last_validation_ok"))
            ),
            last_validation_detail=str(payload.get("last_validation_detail", "")),
            last_validated_at=float(payload.get("last_validated_at", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def write_managed_runtime_state(state: ManagedLlamaCppRuntimeState) -> None:
    paths = gui_managed_paths()
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "release_tag": state.release_tag,
        "variant_id": state.variant_id,
        "install_dir": str(state.install_dir),
        "binary_path": str(state.binary_path),
        "preferred_source": state.preferred_source,
        "installed_at": state.installed_at,
        "last_validation_ok": state.last_validation_ok,
        "last_validation_detail": state.last_validation_detail,
        "last_validated_at": state.last_validated_at,
    }
    paths.runtime_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_managed_runtime_validation(
    *,
    ok: bool,
    detail: str,
    binary_path: Path | None = None,
) -> None:
    state = load_managed_runtime_state()
    if state is None:
        return
    if binary_path is not None and state.binary_path.resolve() != binary_path.expanduser().resolve():
        return
    write_managed_runtime_state(
        replace(
            state,
            last_validation_ok=ok,
            last_validation_detail=detail,
            last_validated_at=time.time(),
        )
    )


def auto_runtime_variant_candidates() -> tuple[str, ...]:
    candidates: list[str] = []
    if _can_load_system_library("nvcuda.dll"):
        candidates.extend(("x64/cuda13", "x64/cuda12"))
    if _can_load_system_library("vulkan-1.dll"):
        candidates.append("x64/vulkan")
    candidates.append(AUTO_RUNTIME_VARIANT_FALLBACK)
    return tuple(dict.fromkeys(candidates))


def fetch_latest_llama_cpp_release(
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> LlamaCppReleaseCatalog:
    downloader = fetch_bytes or _fetch_url_bytes
    raw_payload = downloader(LLAMA_CPP_LATEST_RELEASE_API_URL)
    payload = json.loads(raw_payload.decode("utf-8"))
    assets = tuple(
        LlamaCppReleaseAsset(
            name=str(item["name"]),
            download_url=str(item["browser_download_url"]),
            size_bytes=int(item.get("size") or 0),
        )
        for item in payload.get("assets", ())
        if isinstance(item, dict)
        and item.get("name")
        and item.get("browser_download_url")
    )
    tag_name = str(payload.get("tag_name") or "").strip()
    if not tag_name:
        raise RuntimeError("latest llama.cpp release metadata did not include a tag name")
    if not assets:
        raise RuntimeError(f"latest llama.cpp release {tag_name} did not expose downloadable assets")
    return LlamaCppReleaseCatalog(tag_name=tag_name, assets=assets)


def resolve_runtime_variant(
    catalog: LlamaCppReleaseCatalog,
    *,
    requested_variant: str = "auto",
) -> str:
    normalized = requested_variant.strip().lower()
    if normalized and normalized != "auto":
        if normalized not in MANUAL_RUNTIME_VARIANTS:
            raise RuntimeError(f"unsupported llama.cpp runtime variant: {requested_variant}")
        select_release_assets(catalog, normalized)
        return normalized

    for candidate in auto_runtime_variant_candidates():
        try:
            select_release_assets(catalog, candidate)
        except RuntimeError:
            continue
        return candidate
    raise RuntimeError("latest llama.cpp release did not expose a supported Windows x64 runtime asset")


def select_release_assets(
    catalog: LlamaCppReleaseCatalog,
    variant_id: str,
) -> tuple[LlamaCppReleaseAsset, ...]:
    primary_pattern = _VARIANT_PRIMARY_PATTERNS.get(variant_id)
    if primary_pattern is None:
        raise RuntimeError(f"unsupported llama.cpp runtime variant: {variant_id}")

    primary_asset = next((asset for asset in catalog.assets if primary_pattern.match(asset.name)), None)
    if primary_asset is None:
        raise RuntimeError(
            f"release {catalog.tag_name} does not include the primary asset for variant {variant_id}"
        )

    companions: list[LlamaCppReleaseAsset] = []
    for pattern in _VARIANT_COMPANION_PATTERNS.get(variant_id, ()):
        companion = next((asset for asset in catalog.assets if pattern.match(asset.name)), None)
        if companion is None:
            raise RuntimeError(
                f"release {catalog.tag_name} does not include a required companion asset for {variant_id}"
            )
        companions.append(companion)

    return (primary_asset, *companions)


def install_managed_llama_cpp_runtime(
    *,
    requested_variant: str = "auto",
    force: bool = False,
    install_prerequisites: bool = True,
    progress_callback: Callable[[str, str, str, float | None], None] | None = None,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> ManagedLlamaCppRuntimeState:
    if os.name != "nt":
        raise RuntimeError("Managed llama.cpp bootstrap is currently supported only on Windows GUI.")

    paths = gui_managed_paths()
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    ensure_managed_runtime_prerequisites(
        install=install_prerequisites,
        download_root=paths.state_dir / "downloads",
        progress_callback=progress_callback,
    )
    existing_state = load_managed_runtime_state()
    _report_progress(progress_callback, "runtime_resolve", "Resolve Runtime", "Querying latest llama.cpp release", 0.05)
    catalog = fetch_latest_llama_cpp_release(fetch_bytes=fetch_bytes)
    variant_id = resolve_runtime_variant(catalog, requested_variant=requested_variant)
    assets = select_release_assets(catalog, variant_id)
    variant_dir = _variant_install_dir(paths.runtime_dir, catalog.tag_name, variant_id)

    if (
        not force
        and existing_state is not None
        and existing_state.binary_path.exists()
        and existing_state.install_dir.exists()
        and existing_state.release_tag == catalog.tag_name
        and existing_state.variant_id == variant_id
    ):
        try:
            validate_llama_server_binary(existing_state.binary_path)
        except RuntimeError:
            _report_progress(
                progress_callback,
                "runtime_validate",
                "Validate Runtime",
                "Existing managed runtime failed validation; reinstalling",
                0.12,
            )
        else:
            return existing_state

    if not force:
        existing_binary = _locate_llama_server_binary(variant_dir)
        if existing_binary is not None:
            try:
                _report_progress(
                    progress_callback,
                    "runtime_validate",
                    "Validate Runtime",
                    "Checking existing installed runtime",
                    0.32,
                )
                validate_llama_server_binary(existing_binary)
            except RuntimeError:
                _report_progress(
                    progress_callback,
                    "runtime_validate",
                    "Validate Runtime",
                    "Installed runtime failed validation; reinstalling",
                    0.36,
                )
                if variant_dir.exists():
                    _safe_rmtree(variant_dir, within=paths.runtime_dir)
            else:
                state = ManagedLlamaCppRuntimeState(
                    release_tag=catalog.tag_name,
                    variant_id=variant_id,
                    install_dir=variant_dir,
                    binary_path=existing_binary,
                    preferred_source=MANAGED_RUNTIME_SOURCE,
                    installed_at=time.time(),
                )
                write_managed_runtime_state(state)
                return state

    stage_dir = Path(tempfile.mkdtemp(prefix="llama-cpp-", dir=str(paths.runtime_dir))).resolve()
    try:
        archives = _download_release_assets(
            assets=assets,
            stage_dir=stage_dir,
            progress_callback=progress_callback,
        )
        if force and variant_dir.exists():
            _safe_rmtree(variant_dir, within=paths.runtime_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        _extract_release_archives(
            archives=archives,
            install_dir=variant_dir,
            progress_callback=progress_callback,
        )
        binary_path = _locate_llama_server_binary(variant_dir)
        if binary_path is None:
            raise RuntimeError(
                f"downloaded llama.cpp runtime for {catalog.tag_name} {variant_id} did not contain llama-server.exe"
            )
        _report_progress(
            progress_callback,
            "runtime_validate",
            "Validate Runtime",
            "Running startup probe",
            0.96,
        )
        validate_llama_server_binary(binary_path)
        state = ManagedLlamaCppRuntimeState(
            release_tag=catalog.tag_name,
            variant_id=variant_id,
            install_dir=variant_dir,
            binary_path=binary_path,
            preferred_source=MANAGED_RUNTIME_SOURCE,
            installed_at=time.time(),
        )
        write_managed_runtime_state(state)
        _report_progress(progress_callback, "runtime_ready", "Runtime Ready", f"{catalog.tag_name} {variant_id}", 1.0)
        return state
    finally:
        if stage_dir.exists():
            _safe_rmtree(stage_dir, within=paths.runtime_dir)


def resolve_gui_runtime_binding(
    *,
    explicit_binary_path: Path | None = None,
) -> GuiRuntimeBinding:
    models_dir = gui_managed_models_dir().resolve()
    explicit = _detect_external_llama_server_path(explicit_binary_path)
    if explicit_binary_path is not None and explicit is not None:
        return GuiRuntimeBinding(
            source=OVERRIDE_RUNTIME_SOURCE,
            binary_path=explicit,
            models_dir=models_dir,
        )

    state = load_managed_runtime_state()
    managed_binary = None
    if state is not None and state.binary_path.exists():
        managed_binary = state.binary_path.resolve()

    external_binary = _detect_external_llama_server_path(None)
    preferred_source = state.preferred_source if state is not None else None

    if preferred_source == EXTERNAL_RUNTIME_SOURCE and external_binary is not None:
        return GuiRuntimeBinding(
            source=EXTERNAL_RUNTIME_SOURCE,
            binary_path=external_binary,
            models_dir=models_dir,
        )
    if managed_binary is not None and state is not None:
        return GuiRuntimeBinding(
            source=MANAGED_RUNTIME_SOURCE,
            binary_path=managed_binary,
            models_dir=models_dir,
            release_tag=state.release_tag,
            variant_id=state.variant_id,
            install_dir=state.install_dir,
        )
    if external_binary is not None:
        return GuiRuntimeBinding(
            source=EXTERNAL_RUNTIME_SOURCE,
            binary_path=external_binary,
            models_dir=models_dir,
        )
    return GuiRuntimeBinding(
        source=MISSING_RUNTIME_SOURCE,
        binary_path=None,
        models_dir=models_dir,
    )


def describe_runtime_binding(binding: GuiRuntimeBinding) -> str:
    if binding.source == MANAGED_RUNTIME_SOURCE:
        parts = ["Managed runtime"]
        if binding.release_tag:
            parts.append(binding.release_tag)
        if binding.variant_id:
            parts.append(f"[{binding.variant_id}]")
        return " ".join(parts)
    if binding.source == OVERRIDE_RUNTIME_SOURCE and binding.binary_path is not None:
        return "Configured runtime"
    if binding.source == EXTERNAL_RUNTIME_SOURCE and binding.binary_path is not None:
        return "External runtime"
    return "Managed runtime missing"


def _detect_external_llama_server_path(explicit: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_path = os.environ.get("ISTOTS_LLAMA_SERVER_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    which = shutil.which("llama-server")
    if which is not None:
        candidates.append(Path(which))
    candidates.append(Path.home() / ".local" / "bin" / "llama-server")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _variant_install_dir(runtime_root: Path, release_tag: str, variant_id: str) -> Path:
    return (runtime_root / release_tag / variant_id.replace("/", "-")).resolve()


def validate_llama_server_binary(binary_path: Path, *, timeout: int = 20) -> None:
    normalized_binary = binary_path.expanduser().resolve()
    missing_prerequisites = missing_managed_runtime_prerequisites()
    if missing_prerequisites:
        raise RuntimeError(format_missing_managed_runtime_prerequisites(missing_prerequisites))
    probes = (
        ("--version",),
        ("--help",),
    )
    failures: list[str] = []
    for probe_args in probes:
        completed = subprocess.run(
            [str(normalized_binary), *probe_args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode == 0:
            return
        failures.append(
            _format_binary_probe_failure(
                probe_args=probe_args,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )
    raise RuntimeError(
        "managed llama.cpp runtime failed startup validation.\n"
        f"Binary: {normalized_binary}\n"
        + "\n".join(failures)
    )


def _fetch_url_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "istots-gui-bootstrap",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _download_release_assets(
    *,
    assets: tuple[LlamaCppReleaseAsset, ...],
    stage_dir: Path,
    progress_callback: Callable[[str, str, str, float | None], None] | None,
) -> tuple[Path, ...]:
    archives: list[Path] = []
    total_bytes = sum(asset.size_bytes for asset in assets if asset.size_bytes > 0)
    completed_bytes = 0
    for asset in assets:
        target_path = (stage_dir / asset.name).resolve()
        request = urllib.request.Request(
            asset.download_url,
            headers={"User-Agent": "istots-gui-bootstrap"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            reported_total = int(response.headers.get("Content-Length") or asset.size_bytes or 0)
            downloaded_bytes = 0
            with target_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 128)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total_bytes > 0:
                        ratio = min(1.0, (completed_bytes + downloaded_bytes) / total_bytes)
                        fraction = 0.10 + (ratio * 0.55)
                    elif reported_total > 0:
                        fraction = 0.10 + (min(downloaded_bytes / reported_total, 1.0) * 0.55)
                    else:
                        fraction = None
                    detail = asset.name
                    if reported_total > 0:
                        detail = f"{asset.name} {_format_megabytes(downloaded_bytes)}/{_format_megabytes(reported_total)}"
                    _report_progress(progress_callback, "runtime_download", "Download Runtime", detail, fraction)
        completed_bytes += downloaded_bytes
        archives.append(target_path)
    return tuple(archives)


def _extract_release_archives(
    *,
    archives: tuple[Path, ...],
    install_dir: Path,
    progress_callback: Callable[[str, str, str, float | None], None] | None,
) -> None:
    total_members = 0
    archive_members: list[tuple[Path, list[zipfile.ZipInfo]]] = []
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            archive_members.append((archive_path, members))
            total_members += max(1, len(members))

    processed_members = 0
    for archive_path, members in archive_members:
        with zipfile.ZipFile(archive_path) as archive:
            member_list = members or [None]
            for member in member_list:
                if member is not None:
                    _safe_extract_member(archive, member, install_dir)
                    processed_members += 1
                else:
                    processed_members += 1
                fraction = 0.70 + ((processed_members / total_members) * 0.25)
                detail = archive_path.name
                if member is not None:
                    detail = f"{archive_path.name} {processed_members}/{total_members}"
                _report_progress(progress_callback, "runtime_extract", "Extract Runtime", detail, fraction)


def _safe_extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path) -> None:
    target = (destination / member.filename).resolve()
    destination_root = destination.resolve()
    if target != destination_root and destination_root not in target.parents:
        raise RuntimeError(f"refusing to extract archive member outside install root: {member.filename}")
    archive.extract(member, destination)


def _locate_llama_server_binary(install_dir: Path) -> Path | None:
    if not install_dir.exists():
        return None
    for path in install_dir.rglob("llama-server.exe"):
        return path.resolve()
    return None


def _safe_rmtree(target: Path, *, within: Path) -> None:
    resolved_target = target.resolve()
    resolved_root = within.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise RuntimeError(f"refusing to delete bootstrap path outside managed runtime root: {resolved_target}")
    shutil.rmtree(resolved_target)


def _can_load_system_library(name: str) -> bool:
    if os.name != "nt":
        return False
    try:
        ctypes.WinDLL(name)
    except OSError:
        return False
    return True


def _report_progress(
    progress_callback: Callable[[str, str, str, float | None], None] | None,
    phase: str,
    headline: str,
    detail: str,
    fraction: float | None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(phase, headline, detail, fraction)


def _format_binary_probe_failure(
    *,
    probe_args: tuple[str, ...],
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    details = [f"Probe {' '.join(probe_args)} failed with exit={returncode}."]
    stdout_text = stdout.strip()
    stderr_text = stderr.strip()
    if stdout_text:
        details.append(f"stdout: {stdout_text[-300:]}")
    if stderr_text:
        details.append(f"stderr: {stderr_text[-300:]}")
    return " ".join(details)


def _format_megabytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MB"


def sys_platform() -> str:
    import sys

    return sys.platform
