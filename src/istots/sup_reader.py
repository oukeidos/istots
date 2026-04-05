from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image

from istots.pgs_engine import PgsEngine, ParsedDisplaySet
from istots.pgs_engine.parser import START_STATES

logger = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 15.0
TEN_SECONDS_PTS = 900_000
ONE_MS_PTS = 90
DEDUPE_MAX_GAP_PTS = ONE_MS_PTS * 2


@dataclass
class SubtitleFrame:
    raw_index: int
    start: timedelta
    end: timedelta
    image: Image.Image


@dataclass
class _CandidateFrame:
    raw_index: int
    start_pts: int
    end_pts: int
    image_hash: int
    pixels: list[list[int]]


def iter_sup_frames(
    input_sup: Path,
    max_items: int | None = None,
    on_total: Callable[[int], None] | None = None,
) -> Iterator[SubtitleFrame]:
    if not input_sup.exists():
        raise FileNotFoundError(f"Input SUP file not found: {input_sup}")

    logger.info("python parser started: %s", input_sup)
    heartbeat_stop, started_at = _start_heartbeat("Python parser")
    try:
        engine = PgsEngine(input_sup)
        display_sets = engine.parse_display_sets(predecode_workers=-1)
        candidates = _build_candidates(display_sets)
        finalized = _finalize_candidates(candidates, max_items=max_items)
        frames = _dedupe_consecutive_identical(finalized, max_gap_pts=DEDUPE_MAX_GAP_PTS)
    finally:
        heartbeat_stop.set()

    elapsed = time.monotonic() - started_at
    logger.info("python parser finished: items=%d elapsed=%.1fs", len(frames), elapsed)

    if on_total is not None:
        on_total(len(frames))

    for frame in frames:
        start_ms = _pts_to_ms(frame.start_pts)
        end_ms = _pts_to_ms(frame.end_pts)
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        yield SubtitleFrame(
            raw_index=frame.raw_index,
            start=timedelta(milliseconds=start_ms),
            end=timedelta(milliseconds=end_ms),
            image=_pixels_to_image(frame.pixels),
        )


def _pixels_to_image(pixels: list[list[int]]) -> Image.Image:
    if not pixels:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    width = len(pixels[0])
    height = len(pixels)
    flattened = bytearray(width * height)
    offset = 0
    for row in pixels:
        flattened[offset : offset + width] = bytes(row)
        offset += width
    return Image.frombytes("L", (width, height), bytes(flattened)).convert("RGB")


def _is_start(display_set: ParsedDisplaySet) -> bool:
    if display_set.pcs is None:
        return False
    return display_set.pcs.composition_state in START_STATES


def _build_candidates(display_sets: list[ParsedDisplaySet]) -> list[_CandidateFrame]:
    if not display_sets:
        return []

    ranges: list[tuple[int, int]] = []
    range_start = 0
    for idx in range(1, len(display_sets)):
        if _is_start(display_sets[idx]):
            ranges.append((range_start, idx))
            range_start = idx
    ranges.append((range_start, len(display_sets)))

    candidates: list[_CandidateFrame] = []
    for begin, end in ranges:
        group = display_sets[begin:end]
        has_pts = False
        start_pts = 2**32 - 1
        end_pts = 0
        for row in group:
            if row.pcs is None:
                continue
            pts = row.pcs.presentation_timestamp
            start_pts = min(start_pts, pts)
            end_pts = max(end_pts, pts)
            has_pts = True
        if not has_pts:
            continue

        decoded: list[list[int]] | None = None
        for row in group:
            if not _is_start(row):
                continue
            if not row.complete or row.decoded_pixels is None:
                continue
            decoded = row.decoded_pixels
            break
        if decoded is None:
            continue

        candidates.append(
            _CandidateFrame(
                raw_index=begin,
                start_pts=start_pts,
                end_pts=end_pts,
                image_hash=_hash_pixels(decoded),
                pixels=decoded,
            )
        )

    return candidates


def _finalize_candidates(
    candidates: list[_CandidateFrame],
    max_items: int | None,
) -> list[_CandidateFrame]:
    finalized: list[_CandidateFrame] = []
    for idx in range(len(candidates)):
        item = candidates[idx]
        start_pts = item.start_pts
        end_pts = item.end_pts

        if end_pts <= start_pts:
            if idx + 1 >= len(candidates):
                continue
            next_item = candidates[idx + 1]
            if start_pts + TEN_SECONDS_PTS < next_item.start_pts:
                continue

            min_end = start_pts + ONE_MS_PTS
            next_adjusted = max(0, next_item.start_pts - ONE_MS_PTS)
            end_pts = max(min_end, next_adjusted)

        finalized.append(
            _CandidateFrame(
                raw_index=item.raw_index,
                start_pts=start_pts,
                end_pts=end_pts,
                image_hash=item.image_hash,
                pixels=item.pixels,
            )
        )

        if max_items is not None and len(finalized) >= max_items:
            break

    return finalized


def _dedupe_consecutive_identical(
    candidates: list[_CandidateFrame],
    max_gap_pts: int,
) -> list[_CandidateFrame]:
    deduped: list[_CandidateFrame] = []
    for candidate in candidates:
        if not deduped:
            deduped.append(candidate)
            continue

        prev = deduped[-1]
        gap = max(0, candidate.start_pts - prev.end_pts)
        if prev.image_hash == candidate.image_hash and gap <= max_gap_pts:
            prev.end_pts = max(prev.end_pts, candidate.end_pts)
        else:
            deduped.append(candidate)
    return deduped


def _hash_pixels(pixels: list[list[int]]) -> int:
    digest = hashlib.blake2b(digest_size=8)
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0
    digest.update(height.to_bytes(4, byteorder="big", signed=False))
    digest.update(width.to_bytes(4, byteorder="big", signed=False))
    for row in pixels:
        digest.update(bytes(row))
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def _pts_to_ms(pts: int) -> int:
    return (pts + 45) // 90


def _start_heartbeat(task_label: str) -> tuple[threading.Event, float]:
    stop_event = threading.Event()
    started_at = time.monotonic()

    def _run() -> None:
        while not stop_event.wait(HEARTBEAT_SECONDS):
            elapsed = time.monotonic() - started_at
            logger.info("%s still running... elapsed=%.1fs", task_label, elapsed)

    thread = threading.Thread(target=_run, daemon=True, name="istots-heartbeat")
    thread.start()
    return stop_event, started_at
