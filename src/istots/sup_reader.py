from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image

from istots.pgs_engine import PgsEngine, hash_gray_pixels, shutdown_predecode_pools
from istots.pgs_engine.assembly import (
    DEDUPE_MAX_GAP_PTS,
    build_frame_candidates,
    build_window_frame_candidates,
    dedupe_consecutive_identical,
    dedupe_window_candidates,
    finalize_candidates,
)

logger = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 15.0


@dataclass
class SubtitleFrame:
    raw_index: int
    start: timedelta
    end: timedelta
    image: Image.Image


@dataclass
class SubtitleWindowFrame:
    raw_index: int
    window_id: int
    left: int
    top: int
    right: int
    bottom: int
    start: timedelta
    end: timedelta
    image: Image.Image


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
        display_sets = engine.parse_display_sets(predecode_workers=-1, include_decoded_pixels=True)
        candidates = build_frame_candidates(display_sets, hash_pixels=hash_gray_pixels)
        finalized = finalize_candidates(candidates, max_items=max_items)
        frames = dedupe_consecutive_identical(finalized, max_gap_pts=DEDUPE_MAX_GAP_PTS)
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


def iter_sup_window_frames(
    input_sup: Path,
    max_items: int | None = None,
    on_total: Callable[[int], None] | None = None,
) -> Iterator[SubtitleWindowFrame]:
    if not input_sup.exists():
        raise FileNotFoundError(f"Input SUP file not found: {input_sup}")

    logger.info("python parser started: %s", input_sup)
    heartbeat_stop, started_at = _start_heartbeat("Python parser")
    try:
        engine = PgsEngine(input_sup)
        display_sets = engine.parse_display_sets(predecode_workers=-1, include_decoded_pixels=False)
        candidates = build_window_frame_candidates(display_sets, hash_pixels=hash_gray_pixels)
        finalized = finalize_candidates(candidates, max_items=max_items)
        frames = dedupe_window_candidates(finalized, max_gap_pts=DEDUPE_MAX_GAP_PTS)
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
        yield SubtitleWindowFrame(
            raw_index=frame.raw_index,
            window_id=frame.window_id,
            left=frame.left,
            top=frame.top,
            right=frame.right,
            bottom=frame.bottom,
            start=timedelta(milliseconds=start_ms),
            end=timedelta(milliseconds=end_ms),
            image=_pixels_to_image(frame.pixels),
        )


def release_parser_predecode_workers() -> None:
    shutdown_predecode_pools()


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


def _pts_to_ms(pts: int) -> int:
    return (pts + 45) // 90


def _start_heartbeat(task_label: str) -> tuple[threading.Event, float]:
    stop_event = threading.Event()
    started_at = time.monotonic()

    def _run() -> None:
        while not stop_event.wait(HEARTBEAT_SECONDS):
            elapsed = time.monotonic() - started_at
            logger.info("%s still running... elapsed=%.1fs", task_label, elapsed)

    thread = threading.Thread(target=_run, name="python-parser-heartbeat", daemon=True)
    thread.start()
    return stop_event, started_at
