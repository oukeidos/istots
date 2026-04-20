from __future__ import annotations

from istots.pgs_engine.assembly import (
    EOF_FALLBACK_PTS,
    FrameCandidate,
    WindowFrameCandidate,
    dedupe_consecutive_identical,
    dedupe_window_candidates,
    finalize_candidates,
)


def test_finalize_candidates_repairs_zero_duration_from_next_start() -> None:
    candidates = [
        FrameCandidate(
            raw_index=0,
            start_pts=100,
            end_pts=100,
            image_hash=1,
            pixels=[[1]],
        ),
        FrameCandidate(
            raw_index=1,
            start_pts=200,
            end_pts=260,
            image_hash=2,
            pixels=[[2]],
        ),
    ]

    finalized = finalize_candidates(candidates, max_items=None)

    assert len(finalized) == 2
    assert finalized[0].start_pts == 100
    assert finalized[0].end_pts == 190
    assert finalized[1].end_pts == 260


def test_finalize_candidates_retains_final_zero_duration_candidate_in_combined_arm() -> None:
    candidates = [
        FrameCandidate(
            raw_index=0,
            start_pts=100,
            end_pts=100,
            image_hash=1,
            pixels=[[1]],
        )
    ]

    finalized = finalize_candidates(candidates, max_items=None)

    assert len(finalized) == 1
    assert finalized[0].start_pts == 100
    assert finalized[0].end_pts == 100 + EOF_FALLBACK_PTS


def test_finalize_candidates_retains_long_gap_zero_duration_candidate_in_combined_arm() -> None:
    candidates = [
        FrameCandidate(
            raw_index=0,
            start_pts=100,
            end_pts=100,
            image_hash=1,
            pixels=[[1]],
        ),
        FrameCandidate(
            raw_index=1,
            start_pts=1_000_100,
            end_pts=1_000_260,
            image_hash=2,
            pixels=[[2]],
        ),
    ]

    finalized = finalize_candidates(candidates, max_items=None)

    assert len(finalized) == 2
    assert finalized[0].raw_index == 0
    assert finalized[0].start_pts == 100
    assert finalized[0].end_pts == 1_000_010
    assert finalized[1].raw_index == 1


def test_dedupe_window_candidates_respects_track_identity() -> None:
    candidates = [
        WindowFrameCandidate(
            raw_index=0,
            window_id=0,
            left=10,
            top=20,
            right=11,
            bottom=21,
            start_pts=100,
            end_pts=200,
            image_hash=1,
            pixels=[[1]],
        ),
        WindowFrameCandidate(
            raw_index=1,
            window_id=1,
            left=10,
            top=20,
            right=11,
            bottom=21,
            start_pts=120,
            end_pts=220,
            image_hash=1,
            pixels=[[1]],
        ),
        WindowFrameCandidate(
            raw_index=2,
            window_id=0,
            left=10,
            top=20,
            right=11,
            bottom=21,
            start_pts=205,
            end_pts=260,
            image_hash=1,
            pixels=[[1]],
        ),
    ]

    deduped = dedupe_window_candidates(candidates, max_gap_pts=180)

    assert len(deduped) == 2
    assert deduped[0].window_id == 0
    assert deduped[0].start_pts == 100
    assert deduped[0].end_pts == 260
    assert deduped[1].window_id == 1
    assert deduped[1].start_pts == 120


def test_dedupe_consecutive_identical_merges_full_surface_sequence() -> None:
    candidates = [
        FrameCandidate(
            raw_index=0,
            start_pts=100,
            end_pts=200,
            image_hash=1,
            pixels=[[1]],
        ),
        FrameCandidate(
            raw_index=1,
            start_pts=205,
            end_pts=260,
            image_hash=1,
            pixels=[[1]],
        ),
        FrameCandidate(
            raw_index=2,
            start_pts=300,
            end_pts=360,
            image_hash=2,
            pixels=[[2]],
        ),
    ]

    deduped = dedupe_consecutive_identical(candidates, max_gap_pts=180)

    assert len(deduped) == 2
    assert deduped[0].raw_index == 0
    assert deduped[0].end_pts == 260
    assert deduped[1].raw_index == 2
