from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
import time
from typing import Any

from PIL import Image

from istots.device import resolve_device
from istots.furigana_mask import build_furigana_masks
from istots.ocr import OCRBackendConfig, OCREngine, create_ocr_backend, normalize_ocr_engine
from istots.srt_writer import SubtitleEntry, write_srt
from istots.sup_reader import iter_sup_window_frames

logger = logging.getLogger(__name__)
TALL_SUBTITLE_RATIO_THRESHOLD = 2.0


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


@dataclass(frozen=True)
class _PreparedOCRInput:
    index: int
    frame: Any
    image: Image.Image


def convert_sup_to_srt(
    input_sup: Path,
    output_srt: Path,
    preferred_device: str = "auto",
    engine: str | OCREngine = OCREngine.HF,
    ocr_mode: str = "default",
    model_id: str = "PaddlePaddle/PaddleOCR-VL-1.5",
    models_dir: Path | None = None,
    max_items: int | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    local_files_only: bool = True,
    enable_furigana_mask: bool = False,
    srt_policy: str = "safe",
    runtime_profile: str = "auto",
    runtime_binary_path: Path | None = None,
    runtime_host: str = "127.0.0.1",
    runtime_port: int | None = None,
    runtime_threads: int | None = None,
    runtime_threads_batch: int | None = None,
    runtime_gpu_layers: int | None = None,
    runtime_no_mmproj_offload: bool | None = None,
    runtime_startup_timeout_sec: float = 120.0,
    verbose: bool = True,
) -> ConversionResult:
    if not input_sup.exists():
        raise FileNotFoundError(f"Input SUP file not found: {input_sup}")

    logger.info("starting conversion: input=%s output=%s", input_sup, output_srt)
    device = resolve_device(preferred_device)
    normalized_engine = normalize_ocr_engine(engine)
    normalized_ocr_mode = _normalize_ocr_mode(ocr_mode)
    if normalized_ocr_mode == "fast" and normalized_engine is not OCREngine.LLAMA_SERVER:
        raise ValueError("fast OCR mode requires the llama-server engine")
    backend_config = OCRBackendConfig(
        engine=normalized_engine,
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
        models_dir=models_dir,
        role="ocr",
        profile=runtime_profile,
        binary_path=runtime_binary_path,
        host=runtime_host,
        port=runtime_port,
        threads=runtime_threads,
        threads_batch=runtime_threads_batch,
        gpu_layers=runtime_gpu_layers,
        no_mmproj_offload=runtime_no_mmproj_offload,
        startup_timeout_sec=runtime_startup_timeout_sec,
    )

    if verbose:
        logger.info("ocr mode: %s", normalized_ocr_mode)
        logger.info("requested OCR batch size: %d", batch_size)
        logger.info("furigana masking: %s", "enabled" if enable_furigana_mask else "disabled")
        logger.info("srt policy: %s", srt_policy)

    if normalized_ocr_mode == "fast":
        return _convert_sup_to_srt_fast(
            input_sup=input_sup,
            output_srt=output_srt,
            preferred_device=preferred_device,
            backend_config=backend_config,
            max_items=max_items,
            batch_size=batch_size,
            enable_furigana_mask=enable_furigana_mask,
            srt_policy=srt_policy,
            verbose=verbose,
        )

    backend = None

    try:
        backend, device = _create_ocr_backends(
            [backend_config],
            preferred_device=preferred_device,
            verbose=verbose,
        )
        if verbose:
            logger.info("using device: %s", device)
        active_backend = backend[0]
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
                    texts = active_backend.recognize_batch(images)
                except Exception as exc:
                    if _is_oom_error(exc) and chunk_size > 1:
                        reduced = max(1, chunk_size // 2)
                        effective_batch_size = min(effective_batch_size, reduced)
                        logger.warning(
                            "OCR OOM at batch=%d. Retrying with batch=%d.",
                            chunk_size,
                            effective_batch_size,
                        )
                        active_backend.clear_device_cache()
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

                    segment = _build_window_text_segment(frame, text)
                    if segment is not None:
                        segments.append(segment)

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
        _close_ocr_backends(backend, verbose=verbose)


def _convert_sup_to_srt_fast(
    *,
    input_sup: Path,
    output_srt: Path,
    preferred_device: str,
    backend_config: OCRBackendConfig,
    max_items: int | None,
    batch_size: int,
    enable_furigana_mask: bool,
    srt_policy: str,
    verbose: bool,
) -> ConversionResult:
    prepared_inputs = _collect_prepared_ocr_inputs(
        input_sup,
        max_items=max_items,
        enable_furigana_mask=enable_furigana_mask,
        verbose=verbose,
    )
    wide_inputs = [item for item in prepared_inputs if not _is_tall_subtitle_image(item.image)]
    tall_inputs = [item for item in prepared_inputs if _is_tall_subtitle_image(item.image)]

    if verbose:
        logger.info(
            "hybrid OCR partitioned rows: non_tall=%d tall=%d threshold=%.1f",
            len(wide_inputs),
            len(tall_inputs),
            TALL_SUBTITLE_RATIO_THRESHOLD,
        )

    backend_specs: list[tuple[str, OCRBackendConfig]] = []
    if wide_inputs:
        backend_specs.append(("ocr-fast", replace(backend_config, role="ocr-fast")))
    if tall_inputs:
        backend_specs.append(("ocr", replace(backend_config, role="ocr")))

    active_backends: list[Any] = []
    device = backend_config.device
    try:
        active_backends, device = _create_ocr_backends(
            [config for _, config in backend_specs],
            preferred_device=preferred_device,
            verbose=verbose,
        )
        if verbose:
            logger.info("using device: %s", device)
        backends_by_role = {
            role: backend for (role, _), backend in zip(backend_specs, active_backends, strict=True)
        }
        recognized_by_index: dict[int, str] = {}

        if wide_inputs:
            if verbose:
                logger.info("running fast OCR branch: non_tall=%d role=ocr-fast", len(wide_inputs))
            wide_texts = _recognize_prepared_inputs(
                wide_inputs,
                backend=backends_by_role["ocr-fast"],
                batch_size=batch_size,
                verbose=verbose,
                branch_label="non-tall-fast",
            )
            for item, text in zip(wide_inputs, wide_texts, strict=True):
                recognized_by_index[item.index] = text

        if tall_inputs:
            if verbose:
                logger.info("running default OCR branch: tall=%d role=ocr", len(tall_inputs))
            tall_texts = _recognize_prepared_inputs(
                tall_inputs,
                backend=backends_by_role["ocr"],
                batch_size=batch_size,
                verbose=verbose,
                branch_label="tall-default",
            )
            for item, text in zip(tall_inputs, tall_texts, strict=True):
                recognized_by_index[item.index] = text

        segments = [
            segment
            for item in prepared_inputs
            if (segment := _build_window_text_segment(item.frame, recognized_by_index.get(item.index, ""))) is not None
        ]
        entries = _build_subtitle_entries(segments, srt_policy=srt_policy)
        write_srt(entries, output_srt)
        if verbose:
            logger.info(
                "conversion finished: processed=%d written=%d output=%s",
                len(prepared_inputs),
                len(entries),
                output_srt,
            )
        return ConversionResult(
            output_srt=output_srt,
            processed_count=len(prepared_inputs),
            written_count=len(entries),
            device_used=device,
        )
    finally:
        _close_ocr_backends(active_backends, verbose=verbose)


def _normalize_ocr_mode(ocr_mode: str) -> str:
    normalized = ocr_mode.strip().lower()
    if normalized in {"default", "fast"}:
        return normalized
    raise ValueError(f"unsupported OCR mode: {ocr_mode!r}")


def _collect_prepared_ocr_inputs(
    input_sup: Path,
    *,
    max_items: int | None,
    enable_furigana_mask: bool,
    verbose: bool,
) -> list[_PreparedOCRInput]:
    total = 0

    def on_total(count: int) -> None:
        nonlocal total
        total = count
        if verbose:
            logger.info("parsed SUP OCR inputs: total=%d", total)

    frames = list(iter_sup_window_frames(input_sup, max_items=max_items, on_total=on_total))
    if enable_furigana_mask:
        if verbose:
            logger.info("building furigana mask statistics by orientation")
        images = [result.image for result in build_furigana_masks([frame.image for frame in frames])]
    else:
        images = [frame.image for frame in frames]
    return [
        _PreparedOCRInput(index=index, frame=frame, image=image)
        for index, (frame, image) in enumerate(zip(frames, images, strict=True))
    ]


def _create_ocr_backends(
    configs: list[OCRBackendConfig],
    *,
    preferred_device: str,
    verbose: bool,
) -> tuple[list[Any], str]:
    if not configs:
        return [], "cpu"

    try:
        return _attempt_create_ocr_backends(configs, verbose=verbose), configs[0].device
    except Exception as exc:
        if preferred_device.lower() == "auto" and configs[0].device == "gpu":
            if verbose:
                logger.warning("GPU init failed (%s). Retrying with CPU.", exc)
            cpu_configs = [replace(config, device="cpu") for config in configs]
            return _attempt_create_ocr_backends(cpu_configs, verbose=verbose), "cpu"
        raise


def _attempt_create_ocr_backends(
    configs: list[OCRBackendConfig],
    *,
    verbose: bool,
) -> list[Any]:
    created: list[Any] = []
    try:
        for config in configs:
            if verbose:
                logger.info(
                    "loading OCR backend: engine=%s role=%s model=%s device=%s",
                    config.engine,
                    config.role,
                    config.model_id,
                    config.device,
                )
            created.append(create_ocr_backend(config))
        return created
    except Exception:
        _close_ocr_backends(created, verbose=False)
        raise


def _close_ocr_backends(backends: list[Any] | None, *, verbose: bool) -> None:
    if not backends:
        return
    for backend in backends:
        if hasattr(backend, "close"):
            try:
                backend.close()
                if verbose:
                    logger.info("OCR backend released from device memory")
            except Exception as exc:
                logger.warning("failed to release OCR backend cleanly: %s", exc)


def _recognize_prepared_inputs(
    prepared_inputs: list[_PreparedOCRInput],
    *,
    backend: Any,
    batch_size: int,
    verbose: bool,
    branch_label: str,
) -> list[str]:
    if not prepared_inputs:
        return []

    recognized: list[str] = []
    processed = 0
    effective_batch_size = max(1, batch_size)
    pending = list(prepared_inputs)
    total = len(prepared_inputs)

    while pending:
        chunk_size = min(effective_batch_size, len(pending))
        chunk = pending[:chunk_size]

        start_index = processed + 1
        end_index = processed + len(chunk)
        if verbose:
            logger.info(
                "OCR batch started: branch=%s %s batch=%d",
                branch_label,
                _progress_span_label(start_index, end_index, total),
                chunk_size,
            )

        ocr_started = time.monotonic()
        try:
            texts = backend.recognize_batch([item.image for item in chunk])
        except Exception as exc:
            if _is_oom_error(exc) and chunk_size > 1:
                reduced = max(1, chunk_size // 2)
                effective_batch_size = min(effective_batch_size, reduced)
                logger.warning(
                    "OCR OOM at branch=%s batch=%d. Retrying with batch=%d.",
                    branch_label,
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

        for text in texts:
            processed += 1
            if verbose:
                state = "accepted" if text else "skipped (empty)"
                logger.info(
                    "OCR finished: branch=%s %s %s",
                    branch_label,
                    _progress_label(processed, total),
                    state,
                )
        recognized.extend(texts)

        if verbose:
            logger.info(
                "OCR batch finished: branch=%s %s batch=%d elapsed=%.2fs",
                branch_label,
                _progress_label(processed, total),
                chunk_size,
                ocr_elapsed,
            )
        pending = pending[chunk_size:]

    return recognized


def _is_tall_subtitle_image(image: Image.Image) -> bool:
    width, height = image.size
    if width <= 0:
        return False
    return (float(height) / float(width)) >= TALL_SUBTITLE_RATIO_THRESHOLD


def _build_window_text_segment(frame: Any, text: str) -> _WindowTextSegment | None:
    if not text:
        return None

    end = frame.end
    if end <= frame.start:
        end = frame.start + timedelta(milliseconds=1)

    return _WindowTextSegment(
        start=frame.start,
        end=end,
        text=text,
        window_id=frame.window_id,
        left=frame.left,
        top=frame.top,
        right=frame.right,
        bottom=frame.bottom,
    )


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
