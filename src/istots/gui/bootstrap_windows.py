from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from istots.frozen_subprocess import sanitized_external_subprocess_runtime
from istots.gui.windows_runtime_allowlist import (
    AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT,
    AUTO_MANAGED_RUNTIME_FAMILY_ORDER,
    MANUAL_MANAGED_RUNTIME_CANDIDATE_LIMIT,
    MANUAL_MANAGED_RUNTIME_VARIANTS,
    allowlisted_runtime_tags_for_variant,
)
from istots.llama_mmproj import default_materialized_mmproj_path
from istots.runtime_diagnostics import append_runtime_diagnostic_event
from istots.runtime_prerequisites import (
    ensure_managed_runtime_prerequisites,
    format_missing_managed_runtime_prerequisites,
    missing_managed_runtime_prerequisites,
)
from istots.windows_subprocess import hidden_windows_subprocess_kwargs

LLAMA_CPP_LATEST_RELEASE_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_CPP_RELEASE_BY_TAG_API_URL_TEMPLATE = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
GUI_MANAGED_ROOT_ENV = "ISTOTS_GUI_MANAGED_ROOT"
GUI_RUNTIME_STATE_FILENAME = "llama_cpp_runtime.json"
GUI_RUNTIME_ATTEMPT_HISTORY_FILENAME = "llama_cpp_runtime_attempt_history.json"
MANAGED_RUNTIME_SOURCE = "managed"
EXTERNAL_RUNTIME_SOURCE = "external"
OVERRIDE_RUNTIME_SOURCE = "override"
MISSING_RUNTIME_SOURCE = "missing"
MANUAL_RUNTIME_VARIANTS = MANUAL_MANAGED_RUNTIME_VARIANTS
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
    runtime_attempt_history_path: Path | None = None


@dataclass(frozen=True)
class LlamaCppReleaseAsset:
    name: str
    download_url: str
    size_bytes: int
    sha256_digest: str | None = None


@dataclass(frozen=True)
class LlamaCppReleaseCatalog:
    tag_name: str
    assets: tuple[LlamaCppReleaseAsset, ...]


@dataclass(frozen=True)
class ManagedRuntimeAttemptRecord:
    release_tag: str
    variant_id: str
    attempt_count: int = 0
    last_attempted_at: float = 0.0
    last_outcome: str = ""
    last_detail: str = ""


@dataclass(frozen=True)
class ManagedRuntimeCandidate:
    release_tag: str
    variant_id: str


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
        runtime_attempt_history_path=state_dir / GUI_RUNTIME_ATTEMPT_HISTORY_FILENAME,
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


def _managed_runtime_attempt_history_path(paths: GuiManagedPaths) -> Path:
    configured_path = paths.runtime_attempt_history_path
    if configured_path is not None:
        return configured_path
    return paths.state_dir / GUI_RUNTIME_ATTEMPT_HISTORY_FILENAME


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


def load_managed_runtime_attempt_history() -> dict[tuple[str, str], ManagedRuntimeAttemptRecord]:
    paths = gui_managed_paths()
    path = _managed_runtime_attempt_history_path(paths)
    try:
        if not path.exists():
            return {}
    except OSError:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries", ())
    if not isinstance(entries, list):
        return {}

    history: dict[tuple[str, str], ManagedRuntimeAttemptRecord] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        try:
            record = ManagedRuntimeAttemptRecord(
                release_tag=str(item["release_tag"]),
                variant_id=str(item["variant_id"]),
                attempt_count=max(0, int(item.get("attempt_count", 0))),
                last_attempted_at=float(item.get("last_attempted_at", 0.0)),
                last_outcome=str(item.get("last_outcome", "")),
                last_detail=str(item.get("last_detail", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue
        history[(record.variant_id, record.release_tag)] = record
    return history


def write_managed_runtime_attempt_history(
    history: dict[tuple[str, str], ManagedRuntimeAttemptRecord],
) -> None:
    paths = gui_managed_paths()
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": [
            {
                "release_tag": record.release_tag,
                "variant_id": record.variant_id,
                "attempt_count": record.attempt_count,
                "last_attempted_at": record.last_attempted_at,
                "last_outcome": record.last_outcome,
                "last_detail": record.last_detail,
            }
            for record in sorted(
                history.values(),
                key=lambda item: (item.variant_id, item.release_tag),
            )
        ],
    }
    _managed_runtime_attempt_history_path(paths).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_managed_runtime_attempt(
    *,
    release_tag: str,
    variant_id: str,
    outcome: str,
    detail: str,
) -> ManagedRuntimeAttemptRecord:
    history = load_managed_runtime_attempt_history()
    key = (variant_id, release_tag)
    current = history.get(key)
    record = ManagedRuntimeAttemptRecord(
        release_tag=release_tag,
        variant_id=variant_id,
        attempt_count=(current.attempt_count if current is not None else 0) + 1,
        last_attempted_at=time.time(),
        last_outcome=outcome,
        last_detail=detail,
    )
    history[key] = record
    write_managed_runtime_attempt_history(history)
    return record


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
        candidates.append("x64/cuda12")
    if _can_load_system_library("vulkan-1.dll"):
        candidates.append("x64/vulkan")
    candidates.append(AUTO_RUNTIME_VARIANT_FALLBACK)
    return tuple(
        dict.fromkeys(
            candidate
            for candidate in candidates
            if allowlisted_runtime_tags_for_variant(candidate)
        )
    )


def _parse_release_catalog_payload(
    raw_payload: bytes,
    *,
    source_label: str,
) -> LlamaCppReleaseCatalog:
    payload = json.loads(raw_payload.decode("utf-8"))
    assets = tuple(
        LlamaCppReleaseAsset(
            name=str(item["name"]),
            download_url=str(item["browser_download_url"]),
            size_bytes=int(item.get("size") or 0),
            sha256_digest=_parse_release_asset_sha256_digest(item.get("digest")),
        )
        for item in payload.get("assets", ())
        if isinstance(item, dict)
        and item.get("name")
        and item.get("browser_download_url")
    )
    tag_name = str(payload.get("tag_name") or "").strip()
    if not tag_name:
        raise RuntimeError(f"{source_label} metadata did not include a tag name")
    if not assets:
        raise RuntimeError(f"{source_label} {tag_name} did not expose downloadable assets")
    return LlamaCppReleaseCatalog(tag_name=tag_name, assets=assets)


def fetch_latest_llama_cpp_release(
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> LlamaCppReleaseCatalog:
    downloader = fetch_bytes or _fetch_url_bytes
    return _parse_release_catalog_payload(
        downloader(LLAMA_CPP_LATEST_RELEASE_API_URL),
        source_label="latest llama.cpp release",
    )


def fetch_llama_cpp_release_by_tag(
    tag_name: str,
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> LlamaCppReleaseCatalog:
    normalized_tag = tag_name.strip()
    if not normalized_tag:
        raise RuntimeError("llama.cpp release tag must not be empty")
    downloader = fetch_bytes or _fetch_url_bytes
    api_url = LLAMA_CPP_RELEASE_BY_TAG_API_URL_TEMPLATE.format(
        tag=urllib.parse.quote(normalized_tag, safe=""),
    )
    return _parse_release_catalog_payload(
        downloader(api_url),
        source_label="allowlisted llama.cpp release",
    )


def _normalize_requested_runtime_variant(requested_variant: str) -> str:
    normalized = requested_variant.strip().lower()
    if not normalized:
        return "auto"
    if normalized == "auto":
        return normalized
    if normalized not in MANUAL_RUNTIME_VARIANTS:
        supported = ", ".join(MANUAL_RUNTIME_VARIANTS)
        raise RuntimeError(
            f"unsupported managed llama.cpp runtime variant: {requested_variant}. "
            f"Supported variants: auto, {supported}"
        )
    return normalized


def _ranked_allowlist_tags_for_variant(
    variant_id: str,
    *,
    attempt_history: dict[tuple[str, str], ManagedRuntimeAttemptRecord],
) -> tuple[str, ...]:
    tags = allowlisted_runtime_tags_for_variant(variant_id)
    ranked = list(enumerate(tags))
    ranked.sort(
        key=lambda item: (
            0 if attempt_history.get((variant_id, item[1])) is None else 1,
            (
                attempt_history[(variant_id, item[1])].attempt_count
                if (variant_id, item[1]) in attempt_history
                else 0
            ),
            item[0],
        )
    )
    return tuple(tag for _, tag in ranked)


def _auto_candidate_plan(
    *,
    variant_candidates: tuple[str, ...],
    attempt_history: dict[tuple[str, str], ManagedRuntimeAttemptRecord],
) -> tuple[ManagedRuntimeCandidate, ...]:
    if not variant_candidates:
        return ()

    if variant_candidates == (AUTO_RUNTIME_VARIANT_FALLBACK,):
        ranked_cpu_tags = _ranked_allowlist_tags_for_variant(
            AUTO_RUNTIME_VARIANT_FALLBACK,
            attempt_history=attempt_history,
        )
        return tuple(
            ManagedRuntimeCandidate(release_tag=tag, variant_id=AUTO_RUNTIME_VARIANT_FALLBACK)
            for tag in ranked_cpu_tags[:AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT]
        )

    cpu_variant = AUTO_RUNTIME_VARIANT_FALLBACK if AUTO_RUNTIME_VARIANT_FALLBACK in variant_candidates else None
    non_cpu_variants = tuple(variant for variant in variant_candidates if variant != cpu_variant)
    ranked_by_variant = {
        variant_id: list(
            _ranked_allowlist_tags_for_variant(
                variant_id,
                attempt_history=attempt_history,
            )
        )
        for variant_id in variant_candidates
    }

    planned: list[ManagedRuntimeCandidate] = []
    non_cpu_limit = AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT
    if cpu_variant is not None and ranked_by_variant.get(cpu_variant):
        non_cpu_limit -= 1

    while len(planned) < non_cpu_limit:
        progressed = False
        for variant_id in non_cpu_variants:
            tags = ranked_by_variant[variant_id]
            if not tags:
                continue
            planned.append(ManagedRuntimeCandidate(release_tag=tags.pop(0), variant_id=variant_id))
            progressed = True
            if len(planned) >= non_cpu_limit:
                break
        if not progressed:
            break

    if cpu_variant is not None and ranked_by_variant.get(cpu_variant) and len(planned) < AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT:
        planned.append(
            ManagedRuntimeCandidate(
                release_tag=ranked_by_variant[cpu_variant][0],
                variant_id=cpu_variant,
            )
        )

    while len(planned) < AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT:
        progressed = False
        for variant_id in variant_candidates:
            tags = ranked_by_variant[variant_id]
            if not tags:
                continue
            next_tag = tags.pop(0)
            candidate = ManagedRuntimeCandidate(release_tag=next_tag, variant_id=variant_id)
            if candidate in planned:
                continue
            planned.append(candidate)
            progressed = True
            if len(planned) >= AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT:
                break
        if not progressed:
            break

    return tuple(planned)


def select_allowlisted_runtime_candidates(
    *,
    requested_variant: str,
    attempt_history: dict[tuple[str, str], ManagedRuntimeAttemptRecord] | None = None,
) -> tuple[ManagedRuntimeCandidate, ...]:
    history = attempt_history or {}
    normalized_variant = _normalize_requested_runtime_variant(requested_variant)
    if normalized_variant != "auto":
        ranked_tags = _ranked_allowlist_tags_for_variant(
            normalized_variant,
            attempt_history=history,
        )
        if not ranked_tags:
            raise RuntimeError(f"managed llama.cpp allowlist is empty for variant {normalized_variant}")
        return tuple(
            ManagedRuntimeCandidate(release_tag=tag, variant_id=normalized_variant)
            for tag in ranked_tags[:MANUAL_MANAGED_RUNTIME_CANDIDATE_LIMIT]
        )

    variant_candidates = tuple(
        variant_id
        for variant_id in AUTO_MANAGED_RUNTIME_FAMILY_ORDER
        if variant_id in auto_runtime_variant_candidates()
    )
    if not variant_candidates:
        raise RuntimeError("managed llama.cpp allowlist did not expose a supported Windows x64 runtime family")
    return _auto_candidate_plan(
        variant_candidates=variant_candidates,
        attempt_history=history,
    )


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

    selected_assets = (primary_asset, *companions)
    missing_digest_assets = tuple(
        asset.name for asset in selected_assets if not _has_release_asset_sha256_digest(asset)
    )
    if missing_digest_assets:
        asset_names = ", ".join(missing_digest_assets)
        raise RuntimeError(
            f"release {catalog.tag_name} does not expose a verified SHA-256 digest for: {asset_names}"
        )

    return selected_assets


class ManagedRuntimeCandidateError(RuntimeError):
    def __init__(self, message: str, *, outcome: str) -> None:
        super().__init__(message)
        self.outcome = outcome


def _managed_runtime_candidate_label(candidate: ManagedRuntimeCandidate) -> str:
    return f"{candidate.release_tag} {candidate.variant_id}"


def _format_managed_runtime_candidate_failures(
    *,
    requested_variant: str,
    failure_lines: list[str],
) -> str:
    detail_lines = "\n".join(failure_lines)
    return (
        "managed llama.cpp runtime installation failed after trying approved candidates.\n"
        f"Requested variant: {requested_variant}\n"
        f"Tried candidates:\n{detail_lines}"
    )


def _candidate_progress_fraction(candidate_index: int, candidate_count: int) -> float:
    if candidate_count <= 1:
        return 0.15
    return 0.15 + (0.6 * (candidate_index - 1) / max(1, candidate_count - 1))


def _install_allowlisted_runtime_candidate(
    candidate: ManagedRuntimeCandidate,
    *,
    paths: GuiManagedPaths,
    existing_state: ManagedLlamaCppRuntimeState | None,
    force: bool,
    progress_callback: Callable[[str, str, str, float | None], None] | None,
    fetch_bytes: Callable[[str], bytes] | None,
    cancel_event: threading.Event | None,
) -> ManagedLlamaCppRuntimeState:
    try:
        catalog = fetch_llama_cpp_release_by_tag(candidate.release_tag, fetch_bytes=fetch_bytes)
    except Exception as exc:
        raise ManagedRuntimeCandidateError(str(exc), outcome="release_lookup_failed") from exc

    try:
        assets = select_release_assets(catalog, candidate.variant_id)
    except Exception as exc:
        raise ManagedRuntimeCandidateError(str(exc), outcome="asset_selection_failed") from exc

    variant_dir = _variant_install_dir(paths.runtime_dir, catalog.tag_name, candidate.variant_id)
    append_runtime_diagnostic_event(
        "managed_runtime_candidate_resolved",
        release_tag=catalog.tag_name,
        variant_id=candidate.variant_id,
        assets=[asset.name for asset in assets],
        variant_dir=variant_dir,
    )

    if (
        not force
        and existing_state is not None
        and existing_state.binary_path.exists()
        and existing_state.install_dir.exists()
        and existing_state.release_tag == catalog.tag_name
        and existing_state.variant_id == candidate.variant_id
    ):
        _report_progress(
            progress_callback,
            "runtime_validate",
            "Validate Runtime",
            f"Checking approved runtime {catalog.tag_name} {candidate.variant_id}",
            None,
        )
        try:
            validate_llama_server_binary(existing_state.binary_path, cancel_event=cancel_event)
        except RuntimeError as exc:
            if existing_state.install_dir.exists():
                _safe_rmtree(existing_state.install_dir, within=paths.runtime_dir)
            raise ManagedRuntimeCandidateError(str(exc), outcome="probe_failed") from exc
        append_runtime_diagnostic_event(
            "managed_runtime_reuse_existing_state",
            release_tag=existing_state.release_tag,
            variant_id=existing_state.variant_id,
            binary_path=existing_state.binary_path,
        )
        return existing_state

    if not force:
        existing_binary = _locate_llama_server_binary(variant_dir)
        if existing_binary is not None:
            _report_progress(
                progress_callback,
                "runtime_validate",
                "Validate Runtime",
                f"Checking installed approved runtime {catalog.tag_name} {candidate.variant_id}",
                None,
            )
            try:
                validate_llama_server_binary(existing_binary, cancel_event=cancel_event)
            except RuntimeError as exc:
                if variant_dir.exists():
                    _safe_rmtree(variant_dir, within=paths.runtime_dir)
                raise ManagedRuntimeCandidateError(str(exc), outcome="probe_failed") from exc
            state = ManagedLlamaCppRuntimeState(
                release_tag=catalog.tag_name,
                variant_id=candidate.variant_id,
                install_dir=variant_dir,
                binary_path=existing_binary,
                preferred_source=MANAGED_RUNTIME_SOURCE,
                installed_at=time.time(),
            )
            write_managed_runtime_state(state)
            append_runtime_diagnostic_event(
                "managed_runtime_reuse_variant_dir",
                release_tag=state.release_tag,
                variant_id=state.variant_id,
                binary_path=state.binary_path,
            )
            return state

    stage_dir = Path(tempfile.mkdtemp(prefix="llama-cpp-", dir=str(paths.runtime_dir))).resolve()
    try:
        try:
            archives = _download_release_assets(
                assets=assets,
                stage_dir=stage_dir,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            raise ManagedRuntimeCandidateError(str(exc), outcome="download_failed") from exc
        if force and variant_dir.exists():
            _safe_rmtree(variant_dir, within=paths.runtime_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        append_runtime_diagnostic_event(
            "managed_runtime_extract_start",
            release_tag=catalog.tag_name,
            variant_id=candidate.variant_id,
            variant_dir=variant_dir,
            archives=archives,
        )
        try:
            _extract_release_archives(
                archives=archives,
                install_dir=variant_dir,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            if variant_dir.exists():
                _safe_rmtree(variant_dir, within=paths.runtime_dir)
            raise ManagedRuntimeCandidateError(str(exc), outcome="extract_failed") from exc
        append_runtime_diagnostic_event(
            "managed_runtime_extract_complete",
            release_tag=catalog.tag_name,
            variant_id=candidate.variant_id,
            variant_dir=variant_dir,
        )
        binary_path = _locate_llama_server_binary(variant_dir)
        if binary_path is None:
            if variant_dir.exists():
                _safe_rmtree(variant_dir, within=paths.runtime_dir)
            raise ManagedRuntimeCandidateError(
                f"downloaded llama.cpp runtime for {catalog.tag_name} {candidate.variant_id} did not contain llama-server.exe",
                outcome="extract_failed",
            )
        _report_progress(
            progress_callback,
            "runtime_validate",
            "Validate Runtime",
            f"Running startup probe for approved runtime {catalog.tag_name} {candidate.variant_id}",
            None,
        )
        try:
            validate_llama_server_binary(binary_path, cancel_event=cancel_event)
        except RuntimeError as exc:
            if variant_dir.exists():
                _safe_rmtree(variant_dir, within=paths.runtime_dir)
            raise ManagedRuntimeCandidateError(str(exc), outcome="probe_failed") from exc
        state = ManagedLlamaCppRuntimeState(
            release_tag=catalog.tag_name,
            variant_id=candidate.variant_id,
            install_dir=variant_dir,
            binary_path=binary_path,
            preferred_source=MANAGED_RUNTIME_SOURCE,
            installed_at=time.time(),
        )
        write_managed_runtime_state(state)
        append_runtime_diagnostic_event(
            "managed_runtime_install_complete",
            release_tag=state.release_tag,
            variant_id=state.variant_id,
            binary_path=state.binary_path,
            install_dir=state.install_dir,
        )
        return state
    finally:
        if stage_dir.exists():
            _safe_rmtree(stage_dir, within=paths.runtime_dir)


def install_managed_llama_cpp_runtime(
    *,
    requested_variant: str = "auto",
    force: bool = False,
    install_prerequisites: bool = True,
    progress_callback: Callable[[str, str, str, float | None], None] | None = None,
    fetch_bytes: Callable[[str], bytes] | None = None,
    cancel_event: threading.Event | None = None,
) -> ManagedLlamaCppRuntimeState:
    if os.name != "nt":
        raise RuntimeError("Managed llama.cpp bootstrap is currently supported only on Windows GUI.")

    append_runtime_diagnostic_event(
        "managed_runtime_install_start",
        requested_variant=requested_variant,
        force=force,
        install_prerequisites=install_prerequisites,
    )
    paths = gui_managed_paths()
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        _raise_if_bootstrap_cancelled(cancel_event, stage="prerequisite check")
        ensure_managed_runtime_prerequisites(
            install=install_prerequisites,
            download_root=paths.state_dir / "downloads",
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        existing_state = load_managed_runtime_state()
        attempt_history = load_managed_runtime_attempt_history()
        _report_progress(
            progress_callback,
            "runtime_resolve",
            "Resolve Runtime",
            "Resolving approved llama.cpp runtime candidates",
            0.05,
        )
        _raise_if_bootstrap_cancelled(cancel_event, stage="release resolution")
        candidates = select_allowlisted_runtime_candidates(
            requested_variant=requested_variant,
            attempt_history=attempt_history,
        )
        append_runtime_diagnostic_event(
            "managed_runtime_candidates_planned",
            requested_variant=requested_variant,
            candidates=[
                {
                    "release_tag": candidate.release_tag,
                    "variant_id": candidate.variant_id,
                }
                for candidate in candidates
            ],
        )
        failure_lines: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            _raise_if_bootstrap_cancelled(cancel_event, stage="runtime candidate selection")
            _report_progress(
                progress_callback,
                "runtime_resolve",
                "Resolve Runtime",
                f"Trying approved runtime {index}/{len(candidates)}: {_managed_runtime_candidate_label(candidate)}",
                _candidate_progress_fraction(index, len(candidates)),
            )
            append_runtime_diagnostic_event(
                "managed_runtime_candidate_start",
                release_tag=candidate.release_tag,
                variant_id=candidate.variant_id,
                candidate_index=index,
                candidate_count=len(candidates),
            )
            try:
                state = _install_allowlisted_runtime_candidate(
                    candidate,
                    paths=paths,
                    existing_state=existing_state,
                    force=force,
                    progress_callback=progress_callback,
                    fetch_bytes=fetch_bytes,
                    cancel_event=cancel_event,
                )
            except ManagedRuntimeCandidateError as exc:
                detail = str(exc).strip() or exc.outcome
                record_managed_runtime_attempt(
                    release_tag=candidate.release_tag,
                    variant_id=candidate.variant_id,
                    outcome=exc.outcome,
                    detail=detail,
                )
                append_runtime_diagnostic_event(
                    "managed_runtime_candidate_failed",
                    release_tag=candidate.release_tag,
                    variant_id=candidate.variant_id,
                    candidate_index=index,
                    candidate_count=len(candidates),
                    outcome=exc.outcome,
                    detail=detail,
                )
                failure_lines.append(f"- {_managed_runtime_candidate_label(candidate)}: {detail}")
                continue
            record_managed_runtime_attempt(
                release_tag=state.release_tag,
                variant_id=state.variant_id,
                outcome="installed_ok",
                detail="runtime validated",
            )
            _report_progress(progress_callback, "runtime_ready", "Runtime Ready", f"{state.release_tag} {state.variant_id}", 1.0)
            return state

        raise RuntimeError(
            _format_managed_runtime_candidate_failures(
                requested_variant=requested_variant,
                failure_lines=failure_lines,
            )
        )
    except Exception as exc:
        append_runtime_diagnostic_event(
            "managed_runtime_install_error",
            requested_variant=requested_variant,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


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


def validate_llama_server_binary(
    binary_path: Path,
    *,
    timeout: int = 20,
    cancel_event: threading.Event | None = None,
) -> None:
    normalized_binary = binary_path.expanduser().resolve()
    append_runtime_diagnostic_event(
        "managed_runtime_validation_start",
        binary_path=normalized_binary,
        timeout=timeout,
    )
    missing_prerequisites = missing_managed_runtime_prerequisites()
    if missing_prerequisites:
        raise RuntimeError(format_missing_managed_runtime_prerequisites(missing_prerequisites))
    probe_args = ("--version",)
    append_runtime_diagnostic_event(
        "managed_runtime_validation_probe_start",
        binary_path=normalized_binary,
        probe_args=probe_args,
    )
    with sanitized_external_subprocess_runtime() as sanitized_env:
        process = subprocess.Popen(
            [str(normalized_binary), *probe_args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_env,
            **hidden_windows_subprocess_kwargs(),
        )
        completed = _wait_for_validation_process(
            process,
            timeout=timeout,
            cancel_event=cancel_event,
        )
    if completed.returncode == 0:
        append_runtime_diagnostic_event(
            "managed_runtime_validation_probe_complete",
            binary_path=normalized_binary,
            probe_args=probe_args,
            returncode=completed.returncode,
        )
        return
    failure = _format_binary_probe_failure(
        probe_args=probe_args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    append_runtime_diagnostic_event(
        "managed_runtime_validation_probe_failed",
        binary_path=normalized_binary,
        probe_args=probe_args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    append_runtime_diagnostic_event(
        "managed_runtime_validation_failed",
        binary_path=normalized_binary,
        failures=[failure],
    )
    raise RuntimeError(
        "managed llama.cpp runtime failed startup validation.\n"
        f"Binary: {normalized_binary}\n"
        f"{failure}"
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
    cancel_event: threading.Event | None = None,
) -> tuple[Path, ...]:
    archives: list[Path] = []
    total_bytes = sum(asset.size_bytes for asset in assets if asset.size_bytes > 0)
    completed_bytes = 0
    for asset in assets:
        _raise_if_bootstrap_cancelled(cancel_event, stage="runtime download")
        target_path = (stage_dir / asset.name).resolve()
        append_runtime_diagnostic_event(
            "managed_runtime_asset_download_start",
            asset_name=asset.name,
            target_path=target_path,
            expected_bytes=asset.size_bytes,
        )
        request = urllib.request.Request(
            asset.download_url,
            headers={"User-Agent": "istots-gui-bootstrap"},
        )
        downloaded_bytes = 0
        reported_total = 0
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                reported_total = int(response.headers.get("Content-Length") or asset.size_bytes or 0)
                with target_path.open("wb") as handle:
                    while True:
                        _raise_if_bootstrap_cancelled(cancel_event, stage="runtime download")
                        chunk = response.read(1024 * 128)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
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
            expected_digest = _required_release_asset_sha256_digest(asset)
            actual_digest = digest.hexdigest()
            if actual_digest != expected_digest:
                raise RuntimeError(
                    "downloaded llama.cpp runtime asset failed SHA-256 verification.\n"
                    f"Asset: {asset.name}\n"
                    f"Expected: {expected_digest}\n"
                    f"Actual: {actual_digest}"
                )
            completed_bytes += downloaded_bytes
            archives.append(target_path)
            append_runtime_diagnostic_event(
                "managed_runtime_asset_download_complete",
                asset_name=asset.name,
                target_path=target_path,
                downloaded_bytes=downloaded_bytes,
                reported_total=reported_total,
                sha256=actual_digest,
            )
        except Exception as exc:
            append_runtime_diagnostic_event(
                "managed_runtime_asset_download_error",
                asset_name=asset.name,
                target_path=target_path,
                downloaded_bytes=downloaded_bytes,
                reported_total=reported_total,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
    return tuple(archives)


def _parse_release_asset_sha256_digest(raw_digest: object) -> str | None:
    if raw_digest is None:
        return None
    digest_text = str(raw_digest).strip()
    if not digest_text:
        return None
    prefix = "sha256:"
    if not digest_text.lower().startswith(prefix):
        return None
    normalized = digest_text[len(prefix) :].strip().lower()
    if len(normalized) != 64 or not all(character in "0123456789abcdef" for character in normalized):
        return None
    return normalized


def _has_release_asset_sha256_digest(asset: LlamaCppReleaseAsset) -> bool:
    return bool(asset.sha256_digest and len(asset.sha256_digest) == 64)


def _required_release_asset_sha256_digest(asset: LlamaCppReleaseAsset) -> str:
    digest = asset.sha256_digest or ""
    if len(digest) != 64:
        raise RuntimeError(f"release asset {asset.name} is missing a verified SHA-256 digest")
    return digest


def _extract_release_archives(
    *,
    archives: tuple[Path, ...],
    install_dir: Path,
    progress_callback: Callable[[str, str, str, float | None], None] | None,
    cancel_event: threading.Event | None = None,
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
                _raise_if_bootstrap_cancelled(cancel_event, stage="runtime extraction")
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


def _wait_for_validation_process(
    process: subprocess.Popen[str],
    *,
    timeout: int,
    cancel_event: threading.Event | None,
) -> subprocess.CompletedProcess[str]:
    deadline = time.monotonic() + timeout
    while True:
        returncode = process.poll()
        if returncode is not None:
            stdout, stderr = process.communicate()
            return subprocess.CompletedProcess(process.args, returncode, stdout, stderr)
        _raise_if_bootstrap_cancelled(cancel_event, stage="runtime validation")
        if time.monotonic() >= deadline:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(0.25)


def _raise_if_bootstrap_cancelled(
    cancel_event: threading.Event | None,
    *,
    stage: str,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError(f"managed runtime bootstrap cancelled during {stage}")


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
