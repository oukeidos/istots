from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

GITHUB_RELEASES_LIST_API_URL_TEMPLATE = (
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100&page={page}"
)
ALLOWLIST_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "istots" / "gui" / "windows_runtime_allowlist.py"
)
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "build" / "windows_runtime_allowlist_automation"
LEDGER_FILENAME = "ledger.json"
RUNS_DIRNAME = "runs"
LATEST_SCAN_SUMMARY_FILENAME = "latest_scan_summary.md"
LATEST_SCAN_SUMMARY_JSON_FILENAME = "latest_scan_summary.json"
LATEST_APPLY_SUMMARY_FILENAME = "latest_apply_summary.md"
LATEST_APPLY_SUMMARY_JSON_FILENAME = "latest_apply_summary.json"
DEFAULT_RELEASE_LOOKBACK_DAYS = 120
DEFAULT_RELEASE_OVERLAP_DAYS = 14
DEFAULT_RELEASE_LIMIT = 40
DEFAULT_PROMOTION_TARGET = 3
DEFAULT_ATTEMPT_BUDGET = 8
DEFAULT_GLOBAL_ATTEMPT_BUDGET = 18
SUPPORTED_VARIANTS = ("x64/cpu", "x64/vulkan", "x64/cuda12")
USER_AGENT = "istots-windows-runtime-allowlist-updater/0.4.7"

_VARIANT_PRIMARY_PATTERNS: dict[str, re.Pattern[str]] = {
    "x64/cpu": re.compile(r"^llama-.*-bin-win-cpu-x64\.zip$", re.IGNORECASE),
    "x64/cuda12": re.compile(r"^llama-.*-bin-win-cuda-12\.\d+-x64\.zip$", re.IGNORECASE),
    "x64/vulkan": re.compile(r"^llama-.*-bin-win-vulkan-x64\.zip$", re.IGNORECASE),
}
_VARIANT_COMPANION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "x64/cuda12": (
        re.compile(r"^llama-.*-cuda-12\.\d+-dlls-x64\.zip$", re.IGNORECASE),
    ),
}


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    download_url: str
    size_bytes: int
    sha256_digest: str | None


@dataclass(frozen=True)
class ReleaseCatalog:
    tag_name: str
    published_at: str
    assets: tuple[ReleaseAsset, ...]


@dataclass
class LedgerEntry:
    release_tag: str
    variant_id: str
    status: str
    detail: str
    release_published_at: str
    attempt_count: int = 0
    first_tested_at: str = ""
    last_tested_at: str = ""


@dataclass
class LedgerState:
    last_scan_started_at: str = ""
    last_scan_completed_at: str = ""
    last_run_id: str = ""
    entries: dict[tuple[str, str], LedgerEntry] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvaluation:
    status: str
    detail: str


@dataclass(frozen=True)
class CandidateAttempt:
    release_tag: str
    variant_id: str
    status: str
    detail: str
    release_published_at: str


@dataclass(frozen=True)
class PlannedCandidate:
    release: ReleaseCatalog
    variant_id: str


@dataclass(frozen=True)
class ScanResult:
    run_id: str
    started_at: str
    completed_at: str
    release_window_start: str
    releases_considered: tuple[str, ...]
    planned_candidates: tuple[PlannedCandidate, ...]
    attempts: tuple[CandidateAttempt, ...]
    pending_before: dict[str, tuple[str, ...]]
    pending_after: dict[str, tuple[str, ...]]
    targets_by_variant: dict[str, int]
    attempt_budget_by_variant: dict[str, int]
    global_attempt_budget: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _variant_slug(variant_id: str) -> str:
    return variant_id.replace("/", "-")


def _parse_release_asset_sha256_digest(raw_digest: object) -> str | None:
    if not isinstance(raw_digest, str):
        return None
    prefix = "sha256:"
    normalized = raw_digest.strip()
    if not normalized.lower().startswith(prefix):
        return None
    digest = normalized[len(prefix) :].strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return None
    return digest


def _http_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_url_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_http_headers())
    with urllib.request.urlopen(request) as response:
        return response.read()


def _parse_release_catalog_item(item: object) -> ReleaseCatalog | None:
    if not isinstance(item, dict):
        return None
    tag_name = str(item.get("tag_name") or "").strip()
    published_at = str(item.get("published_at") or "").strip()
    if not tag_name or not published_at:
        return None
    if bool(item.get("draft")) or bool(item.get("prerelease")):
        return None
    assets = []
    for asset in item.get("assets", ()):
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").strip()
        download_url = str(asset.get("browser_download_url") or "").strip()
        if not name or not download_url:
            continue
        assets.append(
            ReleaseAsset(
                name=name,
                download_url=download_url,
                size_bytes=int(asset.get("size") or 0),
                sha256_digest=_parse_release_asset_sha256_digest(asset.get("digest")),
            )
        )
    return ReleaseCatalog(
        tag_name=tag_name,
        published_at=published_at,
        assets=tuple(assets),
    )


def fetch_recent_release_catalogs(
    *,
    release_window_start: datetime,
    release_limit: int,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> tuple[ReleaseCatalog, ...]:
    downloader = fetch_bytes or fetch_url_bytes
    catalogs: list[ReleaseCatalog] = []
    page = 1
    while len(catalogs) < release_limit:
        payload = json.loads(
            downloader(GITHUB_RELEASES_LIST_API_URL_TEMPLATE.format(page=page)).decode("utf-8")
        )
        if not isinstance(payload, list) or not payload:
            break
        stop = False
        for item in payload:
            catalog = _parse_release_catalog_item(item)
            if catalog is None:
                continue
            published_at = _parse_timestamp(catalog.published_at)
            if published_at < release_window_start:
                stop = True
                continue
            catalogs.append(catalog)
            if len(catalogs) >= release_limit:
                break
        if stop or len(payload) < 100:
            break
        page += 1
    return tuple(catalogs[:release_limit])


def load_allowlist_from_source(path: Path) -> dict[str, tuple[str, ...]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id != "WINDOWS_RUNTIME_ALLOWLIST_BY_VARIANT":
                continue
            if node.value is None:
                break
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            return {
                str(variant): tuple(str(tag) for tag in tags)
                for variant, tags in value.items()
                if isinstance(variant, str)
                and isinstance(tags, tuple)
            }
    raise RuntimeError(f"could not load WINDOWS_RUNTIME_ALLOWLIST_BY_VARIANT from {path}")


def _find_allowlist_assignment_span(source_text: str, source_path: Path) -> tuple[int, int]:
    tree = ast.parse(source_text, filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "WINDOWS_RUNTIME_ALLOWLIST_BY_VARIANT":
                if node.end_lineno is None:
                    raise RuntimeError("allowlist assignment did not expose an end line")
                return node.lineno - 1, node.end_lineno
    raise RuntimeError(f"could not find allowlist assignment in {source_path}")


def render_allowlist_assignment(
    allowlist_by_variant: dict[str, tuple[str, ...]],
    *,
    variant_order: tuple[str, ...] | None = None,
) -> str:
    ordered_variants = variant_order or tuple(allowlist_by_variant.keys())
    lines = ["WINDOWS_RUNTIME_ALLOWLIST_BY_VARIANT: dict[str, tuple[str, ...]] = {"]
    for variant_id in ordered_variants:
        tags = allowlist_by_variant.get(variant_id, ())
        lines.append(f'    "{variant_id}": (')
        for tag in tags:
            lines.append(f'        "{tag}",')
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def write_allowlist_to_source(
    path: Path,
    allowlist_by_variant: dict[str, tuple[str, ...]],
) -> None:
    source_text = path.read_text(encoding="utf-8")
    start_line, end_line = _find_allowlist_assignment_span(source_text, path)
    current_allowlist = load_allowlist_from_source(path)
    variant_order = tuple(current_allowlist.keys())
    replacement = render_allowlist_assignment(allowlist_by_variant, variant_order=variant_order)
    source_lines = source_text.splitlines(keepends=True)
    replacement_lines = [line + "\n" for line in replacement.splitlines()]
    source_lines[start_line:end_line] = replacement_lines
    path.write_text("".join(source_lines), encoding="utf-8")


def _dedupe_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tags))


def merge_promotions_into_allowlist(
    allowlist_by_variant: dict[str, tuple[str, ...]],
    promotions_by_variant: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}
    for variant_id, current_tags in allowlist_by_variant.items():
        merged[variant_id] = _dedupe_tags(promotions_by_variant.get(variant_id, ()) + current_tags)
    return merged


def load_ledger(path: Path) -> LedgerState:
    if not path.exists():
        return LedgerState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return LedgerState()
    entries: dict[tuple[str, str], LedgerEntry] = {}
    for item in payload.get("entries", ()):
        if not isinstance(item, dict):
            continue
        try:
            entry = LedgerEntry(
                release_tag=str(item["release_tag"]),
                variant_id=str(item["variant_id"]),
                status=str(item["status"]),
                detail=str(item.get("detail", "")),
                release_published_at=str(item["release_published_at"]),
                attempt_count=max(0, int(item.get("attempt_count", 0))),
                first_tested_at=str(item.get("first_tested_at", "")),
                last_tested_at=str(item.get("last_tested_at", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue
        entries[(entry.variant_id, entry.release_tag)] = entry
    return LedgerState(
        last_scan_started_at=str(payload.get("last_scan_started_at", "")),
        last_scan_completed_at=str(payload.get("last_scan_completed_at", "")),
        last_run_id=str(payload.get("last_run_id", "")),
        entries=entries,
    )


def write_ledger(path: Path, ledger: LedgerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "last_scan_started_at": ledger.last_scan_started_at,
        "last_scan_completed_at": ledger.last_scan_completed_at,
        "last_run_id": ledger.last_run_id,
        "entries": [
            {
                "release_tag": entry.release_tag,
                "variant_id": entry.variant_id,
                "status": entry.status,
                "detail": entry.detail,
                "release_published_at": entry.release_published_at,
                "attempt_count": entry.attempt_count,
                "first_tested_at": entry.first_tested_at,
                "last_tested_at": entry.last_tested_at,
            }
            for entry in sorted(
                ledger.entries.values(),
                key=lambda item: (item.variant_id, item.release_published_at, item.release_tag),
                reverse=False,
            )
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_release_window_start(
    *,
    now_utc: datetime,
    last_scan_completed_at: str,
    lookback_days: int,
    overlap_days: int,
) -> datetime:
    absolute_floor = now_utc - timedelta(days=lookback_days)
    if not last_scan_completed_at:
        return absolute_floor
    last_completed = _parse_timestamp(last_scan_completed_at)
    overlapped = last_completed - timedelta(days=overlap_days)
    return max(absolute_floor, overlapped)


def _find_release_asset(
    assets: tuple[ReleaseAsset, ...],
    pattern: re.Pattern[str],
) -> ReleaseAsset | None:
    return next((asset for asset in assets if pattern.match(asset.name)), None)


def select_release_assets(
    release: ReleaseCatalog,
    variant_id: str,
) -> tuple[ReleaseAsset, ...]:
    primary_pattern = _VARIANT_PRIMARY_PATTERNS.get(variant_id)
    if primary_pattern is None:
        raise RuntimeError(f"unsupported runtime variant: {variant_id}")
    primary_asset = _find_release_asset(release.assets, primary_pattern)
    if primary_asset is None:
        raise RuntimeError(
            f"release {release.tag_name} does not include the primary asset for {variant_id}"
        )

    companions: list[ReleaseAsset] = []
    for pattern in _VARIANT_COMPANION_PATTERNS.get(variant_id, ()):
        companion = _find_release_asset(release.assets, pattern)
        if companion is None:
            raise RuntimeError(
                f"release {release.tag_name} does not include a required companion asset for {variant_id}"
            )
        companions.append(companion)

    selected = (primary_asset, *companions)
    missing_digest_assets = tuple(asset.name for asset in selected if not asset.sha256_digest)
    if missing_digest_assets:
        asset_names = ", ".join(missing_digest_assets)
        raise RuntimeError(
            f"release {release.tag_name} does not expose a verified SHA-256 digest for: {asset_names}"
        )
    return selected


def _candidate_sort_key(entry: LedgerEntry) -> tuple[datetime, datetime, str]:
    published_at = _parse_timestamp(entry.release_published_at)
    last_tested = _parse_timestamp(entry.last_tested_at or entry.release_published_at)
    return (published_at, last_tested, entry.release_tag)


def pending_promotion_entries(
    ledger: LedgerState,
    current_allowlist: dict[str, tuple[str, ...]],
) -> dict[str, tuple[LedgerEntry, ...]]:
    pending: dict[str, list[LedgerEntry]] = {variant_id: [] for variant_id in SUPPORTED_VARIANTS}
    for entry in ledger.entries.values():
        if entry.variant_id not in pending:
            continue
        if entry.status != "passed":
            continue
        if entry.release_tag in current_allowlist.get(entry.variant_id, ()):
            continue
        pending[entry.variant_id].append(entry)
    return {
        variant_id: tuple(
            sorted(entries, key=_candidate_sort_key, reverse=True)
        )
        for variant_id, entries in pending.items()
    }


def pending_promotion_tags(
    ledger: LedgerState,
    current_allowlist: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    return {
        variant_id: tuple(entry.release_tag for entry in entries)
        for variant_id, entries in pending_promotion_entries(ledger, current_allowlist).items()
    }


def build_candidate_queue(
    *,
    releases: tuple[ReleaseCatalog, ...],
    ledger: LedgerState,
    current_allowlist: dict[str, tuple[str, ...]],
    targets_by_variant: dict[str, int],
    attempt_budget_by_variant: dict[str, int],
    global_attempt_budget: int,
) -> tuple[PlannedCandidate, ...]:
    pending = pending_promotion_tags(ledger, current_allowlist)
    remaining_targets = {
        variant_id: max(0, targets_by_variant[variant_id] - len(pending.get(variant_id, ())))
        for variant_id in SUPPORTED_VARIANTS
    }
    attempts_used = {variant_id: 0 for variant_id in SUPPORTED_VARIANTS}
    global_attempts_used = 0
    planned: list[PlannedCandidate] = []
    for release in releases:
        if all(count <= 0 for count in remaining_targets.values()):
            break
        if global_attempts_used >= global_attempt_budget:
            break
        for variant_id in SUPPORTED_VARIANTS:
            if remaining_targets[variant_id] <= 0:
                continue
            if attempts_used[variant_id] >= attempt_budget_by_variant[variant_id]:
                continue
            if release.tag_name in current_allowlist.get(variant_id, ()):
                continue
            if (variant_id, release.tag_name) in ledger.entries:
                continue
            try:
                select_release_assets(release, variant_id)
            except RuntimeError:
                continue
            planned.append(PlannedCandidate(release=release, variant_id=variant_id))
            attempts_used[variant_id] += 1
            global_attempts_used += 1
            if global_attempts_used >= global_attempt_budget:
                break
    return tuple(planned)


def _truncate_output(text: str, *, limit: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _download_asset(asset: ReleaseAsset, destination_dir: Path, *, fetch_bytes: Callable[[str], bytes]) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    payload = fetch_bytes(asset.download_url)
    target_path = destination_dir / asset.name
    target_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest().lower()
    if asset.sha256_digest and digest != asset.sha256_digest:
        raise RuntimeError(
            f"downloaded {asset.name} hash mismatch (expected {asset.sha256_digest}, got {digest})"
        )
    return target_path


def _extract_archives(archives: tuple[Path, ...], install_dir: Path) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(install_dir)


def _locate_llama_server_binary(install_dir: Path) -> Path | None:
    matches = sorted(install_dir.rglob("llama-server.exe"))
    if not matches:
        return None
    return matches[0].resolve()


def _probe_binary(binary_path: Path) -> CandidateEvaluation:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        return CandidateEvaluation(status="probe_failed", detail=f"Probe launch failed: {exc}")
    except subprocess.TimeoutExpired as exc:
        return CandidateEvaluation(
            status="probe_failed",
            detail=f"Probe --version timed out after {exc.timeout} seconds.",
        )
    except OSError as exc:
        return CandidateEvaluation(status="probe_failed", detail=f"Probe launch failed: {exc}")

    if completed.returncode != 0:
        detail_parts = [f"Probe --version failed with exit={completed.returncode}."]
        if completed.stdout:
            detail_parts.append(f"stdout: {_truncate_output(completed.stdout)}")
        if completed.stderr:
            detail_parts.append(f"stderr: {_truncate_output(completed.stderr)}")
        return CandidateEvaluation(status="probe_failed", detail=" ".join(detail_parts))

    return CandidateEvaluation(
        status="passed",
        detail=_truncate_output(completed.stdout or completed.stderr or "Probe --version succeeded."),
    )


def evaluate_release_candidate(
    release: ReleaseCatalog,
    variant_id: str,
    *,
    work_dir: Path,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> CandidateEvaluation:
    downloader = fetch_bytes or fetch_url_bytes
    candidate_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{release.tag_name}-{_variant_slug(variant_id)}-",
            dir=str(work_dir),
        )
    )
    try:
        assets = select_release_assets(release, variant_id)
        archives = tuple(
            _download_asset(asset, candidate_dir / "downloads", fetch_bytes=downloader)
            for asset in assets
        )
        install_dir = candidate_dir / "install"
        _extract_archives(archives, install_dir)
        binary_path = _locate_llama_server_binary(install_dir)
        if binary_path is None:
            return CandidateEvaluation(
                status="extract_failed",
                detail=f"{release.tag_name} {variant_id} did not contain llama-server.exe",
            )
        return _probe_binary(binary_path)
    except RuntimeError as exc:
        detail = str(exc).strip() or type(exc).__name__
        status = "download_failed"
        if "primary asset" in detail or "companion asset" in detail:
            status = "asset_selection_failed"
        elif "SHA-256 digest" in detail or "hash mismatch" in detail:
            status = "download_failed"
        elif "zip" in detail.lower():
            status = "extract_failed"
        return CandidateEvaluation(status=status, detail=detail)
    except zipfile.BadZipFile as exc:
        return CandidateEvaluation(status="extract_failed", detail=f"Invalid zip archive: {exc}")
    finally:
        shutil.rmtree(candidate_dir, ignore_errors=True)


def _record_candidate_attempt(
    ledger: LedgerState,
    *,
    release: ReleaseCatalog,
    variant_id: str,
    evaluation: CandidateEvaluation,
    tested_at: str,
) -> LedgerEntry:
    key = (variant_id, release.tag_name)
    current = ledger.entries.get(key)
    entry = LedgerEntry(
        release_tag=release.tag_name,
        variant_id=variant_id,
        status=evaluation.status,
        detail=evaluation.detail,
        release_published_at=release.published_at,
        attempt_count=(current.attempt_count if current is not None else 0) + 1,
        first_tested_at=current.first_tested_at if current is not None else tested_at,
        last_tested_at=tested_at,
    )
    ledger.entries[key] = entry
    return entry


def execute_scan(
    *,
    releases: tuple[ReleaseCatalog, ...],
    ledger: LedgerState,
    current_allowlist: dict[str, tuple[str, ...]],
    artifact_dir: Path,
    targets_by_variant: dict[str, int],
    attempt_budget_by_variant: dict[str, int],
    global_attempt_budget: int,
    release_window_start: datetime,
    now_utc: datetime,
    evaluate_candidate: Callable[[ReleaseCatalog, str, Path], CandidateEvaluation],
) -> ScanResult:
    run_id = now_utc.strftime("%Y%m%d-%H%M%S")
    started_at = _format_timestamp(now_utc)
    ledger.last_scan_started_at = started_at
    ledger.last_run_id = run_id
    pending_before = pending_promotion_tags(ledger, current_allowlist)
    planned_candidates = build_candidate_queue(
        releases=releases,
        ledger=ledger,
        current_allowlist=current_allowlist,
        targets_by_variant=targets_by_variant,
        attempt_budget_by_variant=attempt_budget_by_variant,
        global_attempt_budget=global_attempt_budget,
    )
    remaining_targets = {
        variant_id: max(0, targets_by_variant[variant_id] - len(pending_before.get(variant_id, ())))
        for variant_id in SUPPORTED_VARIANTS
    }

    run_dir = artifact_dir / RUNS_DIRNAME / run_id
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[CandidateAttempt] = []
    try:
        for planned in planned_candidates:
            if all(count <= 0 for count in remaining_targets.values()):
                break
            if remaining_targets.get(planned.variant_id, 0) <= 0:
                continue
            try:
                evaluation = evaluate_candidate(planned.release, planned.variant_id, work_dir)
            except Exception as exc:
                evaluation = CandidateEvaluation(
                    status="probe_failed",
                    detail=f"Unexpected scan failure: {exc}",
                )
            tested_at = _format_timestamp(_utc_now())
            _record_candidate_attempt(
                ledger,
                release=planned.release,
                variant_id=planned.variant_id,
                evaluation=evaluation,
                tested_at=tested_at,
            )
            attempts.append(
                CandidateAttempt(
                    release_tag=planned.release.tag_name,
                    variant_id=planned.variant_id,
                    status=evaluation.status,
                    detail=evaluation.detail,
                    release_published_at=planned.release.published_at,
                )
            )
            if evaluation.status == "passed" and planned.release.tag_name not in current_allowlist.get(
                planned.variant_id, ()
            ):
                remaining_targets[planned.variant_id] = max(0, remaining_targets[planned.variant_id] - 1)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    completed_at = _format_timestamp(_utc_now())
    ledger.last_scan_completed_at = completed_at
    pending_after = pending_promotion_tags(ledger, current_allowlist)
    return ScanResult(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        release_window_start=_format_timestamp(release_window_start),
        releases_considered=tuple(release.tag_name for release in releases),
        planned_candidates=planned_candidates,
        attempts=tuple(attempts),
        pending_before=pending_before,
        pending_after=pending_after,
        targets_by_variant=dict(targets_by_variant),
        attempt_budget_by_variant=dict(attempt_budget_by_variant),
        global_attempt_budget=global_attempt_budget,
    )


def render_scan_summary(result: ScanResult) -> str:
    lines = [
        "# Windows Runtime Allowlist Scan",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Started: `{result.started_at}`",
        f"- Completed: `{result.completed_at}`",
        f"- Release window start: `{result.release_window_start}`",
        f"- Releases considered: `{len(result.releases_considered)}`",
        f"- Planned attempts: `{len(result.planned_candidates)}`",
        f"- Executed attempts: `{len(result.attempts)}`",
        f"- Global attempt budget: `{result.global_attempt_budget}`",
        "",
        "## Targets",
        "",
    ]
    for variant_id in SUPPORTED_VARIANTS:
        lines.append(
            f"- `{variant_id}`: target `{result.targets_by_variant[variant_id]}`, "
            f"attempt budget `{result.attempt_budget_by_variant[variant_id]}`, "
            f"pending before `{len(result.pending_before.get(variant_id, ()))}`, "
            f"pending after `{len(result.pending_after.get(variant_id, ()))}`"
        )
    lines.extend(["", "## Pending Promotions", ""])
    for variant_id in SUPPORTED_VARIANTS:
        pending = result.pending_after.get(variant_id, ())
        if pending:
            lines.append(f"- `{variant_id}`: {', '.join(f'`{tag}`' for tag in pending)}")
        else:
            lines.append(f"- `{variant_id}`: none")
    lines.extend(["", "## Attempts", ""])
    if not result.attempts:
        lines.append("- No new candidate attempts were needed.")
    else:
        for attempt in result.attempts:
            lines.append(
                f"- `{attempt.release_tag}` `{attempt.variant_id}` -> `{attempt.status}`: {attempt.detail}"
            )
    return "\n".join(lines) + "\n"


def cleanup_stale_work_dirs(artifact_dir: Path) -> None:
    runs_dir = artifact_dir / RUNS_DIRNAME
    if not runs_dir.exists():
        return
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        shutil.rmtree(run_dir / "work", ignore_errors=True)


def write_scan_artifacts(artifact_dir: Path, result: ScanResult) -> None:
    run_dir = artifact_dir / RUNS_DIRNAME / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_markdown = render_scan_summary(result)
    summary_payload = {
        "run_id": result.run_id,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "release_window_start": result.release_window_start,
        "releases_considered": list(result.releases_considered),
        "planned_candidates": [
            {"release_tag": candidate.release.tag_name, "variant_id": candidate.variant_id}
            for candidate in result.planned_candidates
        ],
        "attempts": [
            {
                "release_tag": attempt.release_tag,
                "variant_id": attempt.variant_id,
                "status": attempt.status,
                "detail": attempt.detail,
                "release_published_at": attempt.release_published_at,
            }
            for attempt in result.attempts
        ],
        "pending_before": {variant_id: list(tags) for variant_id, tags in result.pending_before.items()},
        "pending_after": {variant_id: list(tags) for variant_id, tags in result.pending_after.items()},
        "targets_by_variant": dict(result.targets_by_variant),
        "attempt_budget_by_variant": dict(result.attempt_budget_by_variant),
        "global_attempt_budget": result.global_attempt_budget,
    }
    (run_dir / "summary.md").write_text(summary_markdown, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (artifact_dir / LATEST_SCAN_SUMMARY_FILENAME).write_text(summary_markdown, encoding="utf-8")
    (artifact_dir / LATEST_SCAN_SUMMARY_JSON_FILENAME).write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )


def select_promotions_to_apply(
    ledger: LedgerState,
    current_allowlist: dict[str, tuple[str, ...]],
    *,
    targets_by_variant: dict[str, int],
) -> dict[str, tuple[str, ...]]:
    pending_entries_by_variant = pending_promotion_entries(ledger, current_allowlist)
    return {
        variant_id: tuple(
            entry.release_tag
            for entry in pending_entries_by_variant.get(variant_id, ())[: targets_by_variant[variant_id]]
        )
        for variant_id in SUPPORTED_VARIANTS
    }


def write_apply_artifacts(
    artifact_dir: Path,
    *,
    targets_by_variant: dict[str, int],
    promotions_by_variant: dict[str, tuple[str, ...]],
    allowlist_path: Path,
) -> None:
    summary_payload = {
        "allowlist_path": str(allowlist_path),
        "targets_by_variant": dict(targets_by_variant),
        "promotions_by_variant": {variant_id: list(tags) for variant_id, tags in promotions_by_variant.items()},
    }
    lines = [
        "# Windows Runtime Allowlist Apply",
        "",
        f"- Allowlist file: `{allowlist_path}`",
        "",
        "## Promotions",
        "",
    ]
    for variant_id in SUPPORTED_VARIANTS:
        tags = promotions_by_variant.get(variant_id, ())
        if tags:
            lines.append(f"- `{variant_id}`: {', '.join(f'`{tag}`' for tag in tags)}")
        else:
            lines.append(f"- `{variant_id}`: none")
    summary_markdown = "\n".join(lines) + "\n"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / LATEST_APPLY_SUMMARY_FILENAME).write_text(summary_markdown, encoding="utf-8")
    (artifact_dir / LATEST_APPLY_SUMMARY_JSON_FILENAME).write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )


def _default_variant_counts(default_value: int) -> dict[str, int]:
    return {variant_id: default_value for variant_id in SUPPORTED_VARIANTS}


def parse_variant_count_overrides(
    raw_values: list[str] | None,
    *,
    default_value: int,
) -> dict[str, int]:
    values = _default_variant_counts(default_value)
    for raw_value in raw_values or ():
        if "=" not in raw_value:
            raise RuntimeError(f"expected VARIANT=COUNT override, got: {raw_value}")
        variant_id, count_text = raw_value.split("=", 1)
        variant_id = variant_id.strip().lower()
        if variant_id not in SUPPORTED_VARIANTS:
            supported = ", ".join(SUPPORTED_VARIANTS)
            raise RuntimeError(f"unsupported variant override {variant_id!r}; expected one of: {supported}")
        try:
            count = int(count_text.strip())
        except ValueError as exc:
            raise RuntimeError(f"invalid count for {variant_id}: {count_text!r}") from exc
        if count < 0:
            raise RuntimeError(f"count for {variant_id} must be non-negative")
        values[variant_id] = count
    return values


def _print_scan_result(result: ScanResult) -> None:
    print(render_scan_summary(result))


def run_scan_command(args: argparse.Namespace) -> int:
    if os.name != "nt":
        raise RuntimeError("scan is supported only on Windows hosts because it validates Windows llama.cpp binaries")

    allowlist_path = Path(args.allowlist_path).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    ledger_path = artifact_dir / LEDGER_FILENAME
    cleanup_stale_work_dirs(artifact_dir)
    current_allowlist = load_allowlist_from_source(allowlist_path)
    ledger = load_ledger(ledger_path)
    targets_by_variant = parse_variant_count_overrides(args.target, default_value=DEFAULT_PROMOTION_TARGET)
    attempt_budget_by_variant = parse_variant_count_overrides(
        args.attempt_budget,
        default_value=DEFAULT_ATTEMPT_BUDGET,
    )
    now_utc = _utc_now()
    release_window_start = compute_release_window_start(
        now_utc=now_utc,
        last_scan_completed_at=ledger.last_scan_completed_at,
        lookback_days=int(args.lookback_days),
        overlap_days=int(args.overlap_days),
    )
    releases = fetch_recent_release_catalogs(
        release_window_start=release_window_start,
        release_limit=int(args.release_limit),
    )
    result = execute_scan(
        releases=releases,
        ledger=ledger,
        current_allowlist=current_allowlist,
        artifact_dir=artifact_dir,
        targets_by_variant=targets_by_variant,
        attempt_budget_by_variant=attempt_budget_by_variant,
        global_attempt_budget=int(args.global_attempt_budget),
        release_window_start=release_window_start,
        now_utc=now_utc,
        evaluate_candidate=lambda release, variant_id, work_dir: evaluate_release_candidate(
            release,
            variant_id,
            work_dir=work_dir,
        ),
    )
    write_ledger(ledger_path, ledger)
    write_scan_artifacts(artifact_dir, result)
    _print_scan_result(result)
    return 0


def run_apply_command(args: argparse.Namespace) -> int:
    allowlist_path = Path(args.allowlist_path).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    ledger = load_ledger(artifact_dir / LEDGER_FILENAME)
    current_allowlist = load_allowlist_from_source(allowlist_path)
    targets_by_variant = parse_variant_count_overrides(args.target, default_value=DEFAULT_PROMOTION_TARGET)
    promotions_by_variant = select_promotions_to_apply(
        ledger,
        current_allowlist,
        targets_by_variant=targets_by_variant,
    )
    if not any(promotions_by_variant.values()):
        write_apply_artifacts(
            artifact_dir,
            targets_by_variant=targets_by_variant,
            promotions_by_variant=promotions_by_variant,
            allowlist_path=allowlist_path,
        )
        print("No pending passing candidates are available to add to the allowlist.")
        return 0

    updated_allowlist = merge_promotions_into_allowlist(current_allowlist, promotions_by_variant)
    write_allowlist_to_source(allowlist_path, updated_allowlist)
    write_apply_artifacts(
        artifact_dir,
        targets_by_variant=targets_by_variant,
        promotions_by_variant=promotions_by_variant,
        allowlist_path=allowlist_path,
    )
    for variant_id in SUPPORTED_VARIANTS:
        tags = promotions_by_variant.get(variant_id, ())
        if tags:
            print(f"{variant_id}: {', '.join(tags)}")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone maintenance tool for scanning recent llama.cpp Windows releases, "
            "recording validation results, and applying passing candidates to the committed allowlist."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan recent llama.cpp Windows releases, validate new candidates, and update the local ledger.",
    )
    scan_parser.add_argument("--allowlist-path", default=str(ALLOWLIST_MODULE_PATH))
    scan_parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    scan_parser.add_argument("--lookback-days", type=int, default=DEFAULT_RELEASE_LOOKBACK_DAYS)
    scan_parser.add_argument("--overlap-days", type=int, default=DEFAULT_RELEASE_OVERLAP_DAYS)
    scan_parser.add_argument("--release-limit", type=int, default=DEFAULT_RELEASE_LIMIT)
    scan_parser.add_argument("--global-attempt-budget", type=int, default=DEFAULT_GLOBAL_ATTEMPT_BUDGET)
    scan_parser.add_argument(
        "--target",
        action="append",
        help="Override per-family promotion target, for example: x64/vulkan=2",
    )
    scan_parser.add_argument(
        "--attempt-budget",
        action="append",
        help="Override per-family attempt budget, for example: x64/cpu=5",
    )
    scan_parser.set_defaults(func=run_scan_command)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply the top pending passing candidates from the local ledger into the committed allowlist.",
    )
    apply_parser.add_argument("--allowlist-path", default=str(ALLOWLIST_MODULE_PATH))
    apply_parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    apply_parser.add_argument(
        "--target",
        action="append",
        help="Override how many pending passing tags to apply per family, for example: x64/cuda12=1",
    )
    apply_parser.set_defaults(func=run_apply_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
