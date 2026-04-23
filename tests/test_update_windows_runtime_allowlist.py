from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
import sys


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "update_windows_runtime_allowlist.py"
    spec = importlib.util.spec_from_file_location("update_windows_runtime_allowlist", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_release(module, *, tag: str, published_at: str, variant_ids: tuple[str, ...]):
    assets = []
    for variant_id in variant_ids:
        if variant_id == "x64/cpu":
            assets.append(
                module.ReleaseAsset(
                    name=f"llama-{tag}-bin-win-cpu-x64.zip",
                    download_url=f"https://example.invalid/{tag}/cpu.zip",
                    size_bytes=1,
                    sha256_digest="0" * 64,
                )
            )
        elif variant_id == "x64/vulkan":
            assets.append(
                module.ReleaseAsset(
                    name=f"llama-{tag}-bin-win-vulkan-x64.zip",
                    download_url=f"https://example.invalid/{tag}/vulkan.zip",
                    size_bytes=1,
                    sha256_digest="1" * 64,
                )
            )
        elif variant_id == "x64/cuda12":
            assets.append(
                module.ReleaseAsset(
                    name=f"llama-{tag}-bin-win-cuda-12.4-x64.zip",
                    download_url=f"https://example.invalid/{tag}/cuda.zip",
                    size_bytes=1,
                    sha256_digest="2" * 64,
                )
            )
            assets.append(
                module.ReleaseAsset(
                    name=f"llama-{tag}-cuda-12.4-dlls-x64.zip",
                    download_url=f"https://example.invalid/{tag}/cuda-dlls.zip",
                    size_bytes=1,
                    sha256_digest="3" * 64,
                )
            )
    return module.ReleaseCatalog(
        tag_name=tag,
        published_at=published_at,
        assets=tuple(assets),
    )


def test_compute_release_window_start_clamps_old_last_scan() -> None:
    module = _load_script_module()
    now_utc = datetime(2026, 4, 23, 0, 0, tzinfo=UTC)

    result = module.compute_release_window_start(
        now_utc=now_utc,
        last_scan_completed_at="2025-01-01T00:00:00Z",
        lookback_days=120,
        overlap_days=14,
    )

    assert result == datetime(2025, 12, 24, 0, 0, tzinfo=UTC)


def test_pending_promotion_tags_skip_allowlisted_and_sort_newest_first() -> None:
    module = _load_script_module()
    ledger = module.LedgerState(
        entries={
            ("x64/cpu", "b100"): module.LedgerEntry(
                release_tag="b100",
                variant_id="x64/cpu",
                status="passed",
                detail="ok",
                release_published_at="2026-04-20T00:00:00Z",
                last_tested_at="2026-04-21T00:00:00Z",
            ),
            ("x64/cpu", "b101"): module.LedgerEntry(
                release_tag="b101",
                variant_id="x64/cpu",
                status="passed",
                detail="ok",
                release_published_at="2026-04-22T00:00:00Z",
                last_tested_at="2026-04-22T00:00:00Z",
            ),
            ("x64/cpu", "b099"): module.LedgerEntry(
                release_tag="b099",
                variant_id="x64/cpu",
                status="probe_failed",
                detail="fail",
                release_published_at="2026-04-19T00:00:00Z",
                last_tested_at="2026-04-19T00:00:00Z",
            ),
        }
    )
    current_allowlist = {
        "x64/cpu": ("b100",),
        "x64/vulkan": (),
        "x64/cuda12": (),
    }

    pending = module.pending_promotion_tags(ledger, current_allowlist)

    assert pending["x64/cpu"] == ("b101",)


def test_build_candidate_queue_skips_known_results_and_families_with_enough_pending_passes() -> None:
    module = _load_script_module()
    releases = (
        _sample_release(module, tag="b103", published_at="2026-04-23T00:00:00Z", variant_ids=("x64/cpu", "x64/vulkan")),
        _sample_release(module, tag="b102", published_at="2026-04-22T00:00:00Z", variant_ids=("x64/cpu", "x64/vulkan")),
        _sample_release(module, tag="b101", published_at="2026-04-21T00:00:00Z", variant_ids=("x64/cpu", "x64/cuda12")),
    )
    ledger = module.LedgerState(
        entries={
            ("x64/cpu", "b100"): module.LedgerEntry(
                release_tag="b100",
                variant_id="x64/cpu",
                status="passed",
                detail="ok",
                release_published_at="2026-04-20T00:00:00Z",
                last_tested_at="2026-04-20T00:00:00Z",
            ),
            ("x64/vulkan", "b102"): module.LedgerEntry(
                release_tag="b102",
                variant_id="x64/vulkan",
                status="probe_failed",
                detail="fail",
                release_published_at="2026-04-22T00:00:00Z",
                last_tested_at="2026-04-22T00:00:00Z",
            ),
        }
    )
    current_allowlist = {
        "x64/cpu": (),
        "x64/vulkan": (),
        "x64/cuda12": (),
    }

    queue = module.build_candidate_queue(
        releases=releases,
        ledger=ledger,
        current_allowlist=current_allowlist,
        targets_by_variant={"x64/cpu": 1, "x64/vulkan": 1, "x64/cuda12": 1},
        attempt_budget_by_variant={"x64/cpu": 2, "x64/vulkan": 2, "x64/cuda12": 2},
        global_attempt_budget=4,
    )

    assert [(candidate.release.tag_name, candidate.variant_id) for candidate in queue] == [
        ("b103", "x64/vulkan"),
        ("b101", "x64/cuda12"),
    ]


def test_execute_scan_stops_after_family_targets_are_met(tmp_path: Path) -> None:
    module = _load_script_module()
    releases = (
        _sample_release(module, tag="b103", published_at="2026-04-23T00:00:00Z", variant_ids=("x64/cpu",)),
        _sample_release(module, tag="b102", published_at="2026-04-22T00:00:00Z", variant_ids=("x64/cpu",)),
        _sample_release(module, tag="b101", published_at="2026-04-21T00:00:00Z", variant_ids=("x64/cpu",)),
    )
    ledger = module.LedgerState()
    current_allowlist = {
        "x64/cpu": (),
        "x64/vulkan": (),
        "x64/cuda12": (),
    }
    attempted: list[tuple[str, str]] = []

    def fake_evaluate(release, variant_id, work_dir):
        attempted.append((release.tag_name, variant_id))
        return module.CandidateEvaluation(status="passed", detail="ok")

    result = module.execute_scan(
        releases=releases,
        ledger=ledger,
        current_allowlist=current_allowlist,
        artifact_dir=tmp_path / "artifacts",
        targets_by_variant={"x64/cpu": 1, "x64/vulkan": 0, "x64/cuda12": 0},
        attempt_budget_by_variant={"x64/cpu": 3, "x64/vulkan": 0, "x64/cuda12": 0},
        global_attempt_budget=3,
        release_window_start=datetime(2026, 1, 1, tzinfo=UTC),
        now_utc=datetime(2026, 4, 23, tzinfo=UTC),
        evaluate_candidate=fake_evaluate,
    )

    assert attempted == [("b103", "x64/cpu")]
    assert result.pending_after["x64/cpu"] == ("b103",)
    assert not (tmp_path / "artifacts" / "runs" / result.run_id / "work").exists()


def test_write_allowlist_to_source_replaces_only_the_allowlist_block(tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "windows_runtime_allowlist.py"
    source_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "MANUAL_MANAGED_RUNTIME_CANDIDATE_LIMIT = 3",
                "AUTO_MANAGED_RUNTIME_CANDIDATE_LIMIT = 4",
                'WINDOWS_RUNTIME_ALLOWLIST_BY_VARIANT: dict[str, tuple[str, ...]] = {',
                '    "x64/cpu": (',
                '        "b100",',
                "    ),",
                '    "x64/cuda12": (),',
                '    "x64/vulkan": (),',
                "}",
                "",
                "MANUAL_MANAGED_RUNTIME_VARIANTS = (",
                '    "x64/cpu",',
                '    "x64/cuda12",',
                '    "x64/vulkan",',
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )

    updated_allowlist = {
        "x64/cpu": ("b200", "b100"),
        "x64/cuda12": ("b150",),
        "x64/vulkan": (),
    }
    module.write_allowlist_to_source(source_path, updated_allowlist)

    written = source_path.read_text(encoding="utf-8")
    assert '"b200",' in written
    assert '"b150",' in written
    assert "MANUAL_MANAGED_RUNTIME_VARIANTS = (" in written


def test_cleanup_stale_work_dirs_removes_old_run_work_directories(tmp_path: Path) -> None:
    module = _load_script_module()
    work_dir = tmp_path / "runs" / "20260423-000000" / "work" / "candidate"
    work_dir.mkdir(parents=True)
    (work_dir / "artifact.zip").write_bytes(b"zip")

    module.cleanup_stale_work_dirs(tmp_path)

    assert not (tmp_path / "runs" / "20260423-000000" / "work").exists()
