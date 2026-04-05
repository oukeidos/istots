from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import time

from istots.device import resolve_device
from istots.furigana_mask import build_furigana_masks
from istots.ocr.hf_backend import HFPaddleOCRVLBackend
from istots.srt_writer import SubtitleEntry, write_srt
from istots.sup_reader import iter_sup_window_frames

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    output_srt: Path
    processed_count: int
    written_count: int
    device_used: str


@dataclass
class _WindowTextSegment:
    start: timedelta
    end: timedelta
    text: str
    window_id: int
    left: int
    top: int
    right: int
    bottom: int


def convert_sup_to_srt(
    input_sup: Path,
    output_srt: Path,
    preferred_device: str = "auto",
    model_id: str = "PaddlePaddle/PaddleOCR-VL-1.5",
    max_items: int | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    local_files_only: bool = True,
    enable_furigana_mask: bool = False,
    srt_policy: str = "safe",
    verbose: bool = True,
) -> ConversionResult:
    if not input_sup.exists():
        raise FileNotFoundError(f"Input SUP file not found: {input_sup}")

    logger.info("starting conversion: input=%s output=%s", input_sup, output_srt)
    device = resolve_device(preferred_device)
    backend = None

    try:
        logger.info("loading OCR model: %s (device=%s)", model_id, device)
        backend = HFPaddleOCRVLBackend(
            model_id=model_id,
            device=device,
            max_new_tokens=max_new_tokens,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if preferred_device == "auto" and device == "cuda":
            if verbose:
                logger.warning("CUDA init failed (%s). Retrying with CPU.", exc)
            device = "cpu"
            logger.info("loading OCR model: %s (device=%s)", model_id, device)
            backend = HFPaddleOCRVLBackend(
                model_id=model_id,
                device=device,
                max_new_tokens=max_new_tokens,
                local_files_only=local_files_only,
            )
        else:
            raise

    if verbose:
        logger.info("using device: %s", device)
        logger.info("requested OCR batch size: %d", batch_size)
        logger.info("furigana masking: %s", "enabled" if enable_furigana_mask else "disabled")
        logger.info("srt policy: %s", srt_policy)

    try:
        segments: list[_WindowTextSegment] = []
        processed = 0
        total = 0
        effective_batch_size = max(1, batch_size)
        frame_buffer = []
        prepared_images = None

        def on_total(count: int) -> None:
            nonlocal total
            total = count
            if verbose:
                logger.info("parsed SUP OCR inputs: total=%d", total)

        def flush_buffer() -> None:
            nonlocal processed, effective_batch_size, frame_buffer
            pending = list(frame_buffer)
            frame_buffer = []

            while pending:
                chunk_size = min(effective_batch_size, len(pending))
                chunk = pending[:chunk_size]

                start_index = processed + 1
                end_index = processed + len(chunk)
                if verbose:
                    logger.info(
                        "OCR batch started: %s batch=%d",
                        _progress_span_label(start_index, end_index, total),
                        chunk_size,
                    )

                ocr_started = time.monotonic()
                try:
                    if prepared_images is not None:
                        start_offset = start_index - 1
                        images = prepared_images[start_offset : start_offset + len(chunk)]
                    else:
                        images = [frame.image for frame in chunk]
                    texts = backend.recognize_batch(images)
                except Exception as exc:
                    if _is_oom_error(exc) and chunk_size > 1:
                        reduced = max(1, chunk_size // 2)
                        effective_batch_size = min(effective_batch_size, reduced)
                        logger.warning(
                            "OCR OOM at batch=%d. Retrying with batch=%d.",
                            chunk_size,
                            effective_batch_size,
                        )
                        backend.clear_device_cache()
                        continue
                    raise

                ocr_elapsed = time.monotonic() - ocr_started
                if len(texts) != len(chunk):
                    raise RuntimeError(
                        "OCR backend returned mismatched result count: "
                        f"expected={len(chunk)} actual={len(texts)}"
                    )

                for frame, text in zip(chunk, texts):
                    processed += 1
                    if not text:
                        if verbose:
                            logger.info("OCR finished: %s skipped (empty)", _progress_label(processed, total))
                        continue

                    end = frame.end
                    if end <= frame.start:
                        end = frame.start + timedelta(milliseconds=1)

                    segments.append(
                        _WindowTextSegment(
                            start=frame.start,
                            end=end,
                            text=text,
                            window_id=frame.window_id,
                            left=frame.left,
                            top=frame.top,
                            right=frame.right,
                            bottom=frame.bottom,
                        )
                    )

                    if verbose:
                        logger.info("OCR finished: %s accepted", _progress_label(processed, total))

                if verbose:
                    logger.info(
                        "OCR batch finished: %s batch=%d elapsed=%.2fs",
                        _progress_label(processed, total),
                        chunk_size,
                        ocr_elapsed,
                    )
                pending = pending[chunk_size:]

        if enable_furigana_mask:
            frames = list(iter_sup_window_frames(input_sup, max_items=max_items, on_total=on_total))
            if verbose:
                logger.info("building furigana mask statistics by orientation")
            prepared_images = [result.image for result in build_furigana_masks([frame.image for frame in frames])]
            frame_iterable = frames
        else:
            frame_iterable = iter_sup_window_frames(input_sup, max_items=max_items, on_total=on_total)

        for frame in frame_iterable:
            frame_buffer.append(frame)
            if len(frame_buffer) >= effective_batch_size:
                flush_buffer()

        if frame_buffer:
            flush_buffer()

        entries = _build_subtitle_entries(segments, srt_policy=srt_policy)
        write_srt(entries, output_srt)
        if verbose:
            logger.info(
                "conversion finished: processed=%d written=%d output=%s",
                processed,
                len(entries),
                output_srt,
            )
        return ConversionResult(
            output_srt=output_srt,
            processed_count=processed,
            written_count=len(entries),
            device_used=device,
        )
    finally:
        if backend is not None and hasattr(backend, "close"):
            try:
                backend.close()
                if verbose:
                    logger.info("OCR backend released from device memory")
            except Exception as exc:
                logger.warning("failed to release OCR backend cleanly: %s", exc)


def _build_subtitle_entries(
    segments: list[_WindowTextSegment],
    srt_policy: str,
) -> list[SubtitleEntry]:
    if srt_policy == "safe":
        return _merge_window_segments(segments)
    if srt_policy == "overlap":
        return _overlap_window_segments(segments)
    raise ValueError(f"unsupported srt policy: {srt_policy}")


def _clean_window_segments(segments: list[_WindowTextSegment]) -> list[_WindowTextSegment]:
    return [
        _WindowTextSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
            window_id=segment.window_id,
            left=segment.left,
            top=segment.top,
            right=segment.right,
            bottom=segment.bottom,
        )
        for segment in segments
        if segment.text.strip() and segment.end > segment.start
    ]


def _merge_window_segments(segments: list[_WindowTextSegment]) -> list[SubtitleEntry]:
    cleaned = _clean_window_segments(segments)
    if not cleaned:
        return []

    boundaries = sorted({segment.start for segment in cleaned} | {segment.end for segment in cleaned})
    entries: list[SubtitleEntry] = []

    for index in range(len(boundaries) - 1):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end <= start:
            continue

        active = [
            segment
            for segment in cleaned
            if segment.start < end and segment.end > start
        ]
        if not active:
            continue

        active.sort(key=lambda segment: (segment.top, segment.left, segment.right, segment.window_id))
        lines: list[str] = []
        seen: set[str] = set()
        for segment in active:
            if segment.text in seen:
                continue
            lines.append(segment.text)
            seen.add(segment.text)
        if not lines:
            continue

        text = "\n".join(lines)
        if entries and entries[-1].end == start and entries[-1].text == text:
            entries[-1].end = end
            continue

        entries.append(
            SubtitleEntry(
                index=len(entries) + 1,
                start=start,
                end=end,
                text=text,
            )
        )

    for index, entry in enumerate(entries, start=1):
        entry.index = index
    return entries


def _overlap_window_segments(segments: list[_WindowTextSegment]) -> list[SubtitleEntry]:
    cleaned = _clean_window_segments(segments)
    cleaned.sort(
        key=lambda segment: (
            segment.start,
            segment.top,
            segment.left,
            segment.right,
            segment.window_id,
            segment.end,
        )
    )
    return [
        SubtitleEntry(
            index=index,
            start=segment.start,
            end=segment.end,
            text=segment.text,
        )
        for index, segment in enumerate(cleaned, start=1)
    ]


def _percent(current: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (current / total) * 100.0


def _progress_label(current: int, total: int) -> str:
    if total > 0:
        return f"{current}/{total} ({_percent(current, total):.1f}%)"
    return f"{current}"


def _progress_span_label(start: int, end: int, total: int) -> str:
    if total > 0:
        return f"{start}-{end}/{total}"
    return f"{start}-{end}"


def _is_oom_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "outofmemory" in name:
        return True

    message = str(exc).lower()
    patterns = (
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "hip error out of memory",
    )
    return any(pattern in message for pattern in patterns)
