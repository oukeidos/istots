from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol, Sequence, TypeVar


ONE_MS_PTS = 90
EOF_FALLBACK_PTS = 180_000
DEDUPE_MAX_GAP_PTS = ONE_MS_PTS * 2


class _PcsLike(Protocol):
    presentation_timestamp: int


class _WindowLike(Protocol):
    window_id: int
    left: int
    top: int
    right: int
    bottom: int
    pixels: list[list[int]]


class _DisplaySetLike(Protocol):
    raw_index: int
    pcs: _PcsLike | None
    complete: bool
    decoded_pixels: list[list[int]] | None
    decoded_windows: Sequence[_WindowLike]


@dataclass
class FrameCandidate:
    raw_index: int
    start_pts: int
    end_pts: int
    image_hash: int
    pixels: list[list[int]]


@dataclass
class WindowFrameCandidate:
    raw_index: int
    window_id: int
    left: int
    top: int
    right: int
    bottom: int
    start_pts: int
    end_pts: int
    image_hash: int
    pixels: list[list[int]]


_T = TypeVar("_T", FrameCandidate, WindowFrameCandidate)


def _complete_rows(display_sets: Sequence[_DisplaySetLike]) -> list[_DisplaySetLike]:
    return [row for row in display_sets if row.pcs is not None and row.complete]


def build_frame_candidates(
    display_sets: Sequence[_DisplaySetLike],
    *,
    hash_pixels: Callable[[list[list[int]]], int],
) -> list[FrameCandidate]:
    complete_rows = _complete_rows(display_sets)
    if not complete_rows:
        return []

    candidates: list[FrameCandidate] = []
    for idx, row in enumerate(complete_rows):
        assert row.pcs is not None
        decoded = row.decoded_pixels
        if decoded is None:
            continue
        start_pts = row.pcs.presentation_timestamp
        if idx + 1 < len(complete_rows):
            next_row = complete_rows[idx + 1]
            assert next_row.pcs is not None
            end_pts = next_row.pcs.presentation_timestamp
        else:
            end_pts = start_pts

        candidates.append(
            FrameCandidate(
                raw_index=row.raw_index,
                start_pts=start_pts,
                end_pts=end_pts,
                image_hash=hash_pixels(decoded),
                pixels=decoded,
            )
        )

    return candidates


def build_window_frame_candidates(
    display_sets: Sequence[_DisplaySetLike],
    *,
    hash_pixels: Callable[[list[list[int]]], int],
) -> list[WindowFrameCandidate]:
    complete_rows = _complete_rows(display_sets)
    if not complete_rows:
        return []

    candidates: list[WindowFrameCandidate] = []
    for idx, row in enumerate(complete_rows):
        assert row.pcs is not None
        decoded_windows = row.decoded_windows
        if not decoded_windows:
            continue
        start_pts = row.pcs.presentation_timestamp
        if idx + 1 < len(complete_rows):
            next_row = complete_rows[idx + 1]
            assert next_row.pcs is not None
            end_pts = next_row.pcs.presentation_timestamp
        else:
            end_pts = start_pts

        for window in decoded_windows:
            candidates.append(
                WindowFrameCandidate(
                    raw_index=row.raw_index,
                    window_id=window.window_id,
                    left=window.left,
                    top=window.top,
                    right=window.right,
                    bottom=window.bottom,
                    start_pts=start_pts,
                    end_pts=end_pts,
                    image_hash=hash_pixels(window.pixels),
                    pixels=window.pixels,
                )
            )

    return candidates


def finalize_candidates(candidates: Sequence[_T], max_items: int | None) -> list[_T]:
    finalized: list[_T] = []
    for idx in range(len(candidates)):
        item = candidates[idx]
        start_pts = item.start_pts
        end_pts = item.end_pts

        if end_pts <= start_pts:
            if idx + 1 >= len(candidates):
                end_pts = start_pts + EOF_FALLBACK_PTS
            else:
                next_item = candidates[idx + 1]
                min_end = start_pts + ONE_MS_PTS
                next_adjusted = max(0, next_item.start_pts - ONE_MS_PTS)
                end_pts = max(min_end, next_adjusted)

        finalized.append(replace(item, end_pts=end_pts))

        if max_items is not None and len(finalized) >= max_items:
            break

    return finalized


def dedupe_consecutive_identical(candidates: Sequence[_T], max_gap_pts: int) -> list[_T]:
    deduped: list[_T] = []

    for candidate in candidates:
        current = replace(candidate)
        if not deduped:
            deduped.append(current)
            continue

        prev = deduped[-1]
        gap = max(0, current.start_pts - prev.end_pts)
        if prev.image_hash == current.image_hash and gap <= max_gap_pts:
            prev.end_pts = max(prev.end_pts, current.end_pts)
        else:
            deduped.append(current)

    return deduped


def dedupe_window_candidates(
    candidates: Sequence[WindowFrameCandidate],
    max_gap_pts: int,
) -> list[WindowFrameCandidate]:
    deduped: list[WindowFrameCandidate] = []
    last_index_by_track: dict[tuple[int, int, int, int, int], int] = {}

    for candidate in candidates:
        current = replace(candidate)
        track_key = (
            current.window_id,
            current.left,
            current.top,
            current.right,
            current.bottom,
        )
        prev_index = last_index_by_track.get(track_key)
        if prev_index is not None:
            prev = deduped[prev_index]
            gap = max(0, current.start_pts - prev.end_pts)
            if prev.image_hash == current.image_hash and gap <= max_gap_pts:
                prev.end_pts = max(prev.end_pts, current.end_pts)
                continue

        deduped.append(current)
        last_index_by_track[track_key] = len(deduped) - 1

    return deduped


__all__ = [
    "DEDUPE_MAX_GAP_PTS",
    "FrameCandidate",
    "WindowFrameCandidate",
    "build_frame_candidates",
    "build_window_frame_candidates",
    "dedupe_consecutive_identical",
    "dedupe_window_candidates",
    "finalize_candidates",
]
