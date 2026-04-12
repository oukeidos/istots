from __future__ import annotations

from contextlib import contextmanager
import logging
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
import time
from typing import Any, Iterator

from PIL import Image

from istots.anchor_merge import apply_union_anchor_merge, build_focus_context
from istots.corrector import (
    LOCAL_QWEN_CTX_SIZE,
    LOCAL_QWEN_MAX_NEW_TOKENS,
    STRICT_OCR_V1_PROMPT,
    ConservativeCorrectionRecord,
    CorrectorConfig,
    CorrectorMode,
    corrector_name_for_config,
    corrector_prompt_for_shape,
    request_gemini_correction,
    write_correction_records,
)
from istots.detector import HybridDetectorRecord, write_hybrid_detector_records
from istots.device import resolve_hf_device
from istots.furigana_mask import build_furigana_masks
from istots.ocr import OCRBackendConfig, OCREngine, create_ocr_backend, normalize_ocr_engine
from istots.srt_writer import SubtitleEntry, write_srt
from istots.sup_reader import iter_sup_window_frames
from istots.text_diff import assess_difference

logger = logging.getLogger(__name__)
TALL_SUBTITLE_RATIO_THRESHOLD = 2.0


@dataclass
class ConversionResult:
    output_srt: Path
    processed_count: int
    written_count: int
    device_used: str
    detector_record_count: int = 0
    correction_record_count: int = 0
    correction_applied_count: int = 0


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
    hf_device: str = "auto",
    hf_dtype: str = "auto",
    engine: str | OCREngine = OCREngine.HF,
    ocr_mode: str = "default",
    detector_output: Path | None = None,
    corrector_config: CorrectorConfig | None = None,
    model_id: str = "PaddlePaddle/PaddleOCR-VL-1.5",
    models_dir: Path | None = None,
    max_items: int | None = None,
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
    normalized_engine = normalize_ocr_engine(engine)
    normalized_ocr_mode = _normalize_ocr_mode(ocr_mode)
    resolved_hf_device = resolve_hf_device(hf_device) if normalized_engine is OCREngine.HF else None
    runtime_label = resolved_hf_device if resolved_hf_device is not None else runtime_profile
    if normalized_ocr_mode == "fast" and normalized_engine is not OCREngine.LLAMA_SERVER:
        raise ValueError("fast OCR mode requires the llama-server engine")
    if detector_output is not None and normalized_engine is not OCREngine.LLAMA_SERVER:
        raise ValueError("detector output requires the llama-server engine")
    if detector_output is not None and normalized_ocr_mode != "default":
        raise ValueError("detector output requires the default OCR mode")
    if corrector_config is not None and normalized_engine is not OCREngine.LLAMA_SERVER:
        raise ValueError("correction requires the llama-server engine")
    if corrector_config is not None and normalized_ocr_mode != "default":
        raise ValueError("correction requires the default OCR mode")
    backend_config = OCRBackendConfig(
        engine=normalized_engine,
        model_id=model_id,
        device=resolved_hf_device,
        hf_dtype=hf_dtype,
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
        logger.info("furigana masking: %s", "enabled" if enable_furigana_mask else "disabled")
        logger.info("srt policy: %s", srt_policy)

    if normalized_ocr_mode == "fast":
        return _convert_sup_to_srt_fast(
            input_sup=input_sup,
            output_srt=output_srt,
            backend_config=backend_config,
            runtime_label=runtime_label,
            max_items=max_items,
            enable_furigana_mask=enable_furigana_mask,
            srt_policy=srt_policy,
            verbose=verbose,
        )
    if detector_output is not None or corrector_config is not None:
        return _convert_sup_to_srt_default_with_detector(
            input_sup=input_sup,
            output_srt=output_srt,
            backend_config=backend_config,
            runtime_label=runtime_label,
            detector_output=detector_output,
            corrector_config=corrector_config,
            max_items=max_items,
            enable_furigana_mask=enable_furigana_mask,
            srt_policy=srt_policy,
            verbose=verbose,
        )

    prepared_inputs = _collect_prepared_ocr_inputs(
        input_sup,
        max_items=max_items,
        enable_furigana_mask=enable_furigana_mask,
        verbose=verbose,
    )
    with _managed_ocr_backend(
        backend_config,
        allow_hf_auto_cpu_fallback=normalized_engine is OCREngine.HF and hf_device == "auto",
        verbose=verbose,
    ) as (active_backend, runtime_used):
        if verbose:
            _log_runtime_selection(engine=normalized_engine, runtime_used=runtime_used)
        texts = _recognize_prepared_inputs(
            prepared_inputs,
            backend=active_backend,
            verbose=verbose,
            branch_label="default",
        )
        segments = [
            segment
            for item, text in zip(prepared_inputs, texts, strict=True)
            if (segment := _build_window_text_segment(item.frame, text)) is not None
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
            device_used=runtime_used,
        )


def _convert_sup_to_srt_fast(
    *,
    input_sup: Path,
    output_srt: Path,
    backend_config: OCRBackendConfig,
    runtime_label: str,
    max_items: int | None,
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

    device_logged = False
    recognized_by_index: dict[int, str] = {}

    for role, config in backend_specs:
        branch_inputs = wide_inputs if role == "ocr-fast" else tall_inputs
        branch_label = "non-tall-fast" if role == "ocr-fast" else "tall-default"
        if not branch_inputs:
            continue
        with _managed_ocr_backend(
            config,
            allow_hf_auto_cpu_fallback=False,
            verbose=verbose,
        ) as (backend, _):
            if verbose and not device_logged:
                _log_runtime_selection(engine=backend_config.engine, runtime_used=runtime_label)
            device_logged = True
            if role == "ocr-fast":
                if verbose:
                    logger.info("running fast OCR branch: non_tall=%d role=ocr-fast", len(branch_inputs))
            elif verbose:
                logger.info("running default OCR branch: tall=%d role=ocr", len(branch_inputs))
            branch_texts = _recognize_prepared_inputs(
                branch_inputs,
                backend=backend,
                verbose=verbose,
                branch_label=branch_label,
            )
            for item, text in zip(branch_inputs, branch_texts, strict=True):
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
        device_used=runtime_label,
    )


def _convert_sup_to_srt_default_with_detector(
    *,
    input_sup: Path,
    output_srt: Path,
    backend_config: OCRBackendConfig,
    runtime_label: str,
    detector_output: Path | None,
    corrector_config: CorrectorConfig | None,
    max_items: int | None,
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
    with _managed_ocr_backend(
        backend_config,
        allow_hf_auto_cpu_fallback=False,
        verbose=verbose,
    ) as (baseline_backend, _):
        if verbose:
            _log_runtime_selection(engine=backend_config.engine, runtime_used=runtime_label)
        baseline_texts = _recognize_prepared_inputs(
            prepared_inputs,
            backend=baseline_backend,
            verbose=verbose,
            branch_label="default",
        )

    detector_records = _build_hybrid_detector_records(
        prepared_inputs,
        baseline_texts,
        detector_backend_config=backend_config,
        verbose=verbose,
    )
    if detector_output is not None:
        write_hybrid_detector_records(detector_output, detector_records)

    correction_records: list[ConservativeCorrectionRecord] = []
    final_texts = list(baseline_texts)
    if corrector_config is not None:
        correction_records = _apply_conservative_corrections(
            prepared_inputs=prepared_inputs,
            detector_records=detector_records,
            corrector_config=corrector_config,
            runtime_profile=backend_config.profile,
            runtime_binary_path=backend_config.binary_path,
            runtime_host=backend_config.host,
            verbose=verbose,
        )
        for record in correction_records:
            final_texts[record.index] = record.conservative_merged_text
        if corrector_config.output_path is not None:
            write_correction_records(corrector_config.output_path, correction_records)

    segments = [
        segment
        for item, text in zip(prepared_inputs, final_texts, strict=True)
        if (segment := _build_window_text_segment(item.frame, text)) is not None
    ]
    entries = _build_subtitle_entries(segments, srt_policy=srt_policy)
    write_srt(entries, output_srt)
    if verbose:
        logger.info(
            "conversion finished: processed=%d written=%d detector_disagreements=%d correction_rows=%d output=%s",
            len(prepared_inputs),
            len(entries),
            len(detector_records),
            len(correction_records),
            output_srt,
        )
    return ConversionResult(
        output_srt=output_srt,
        processed_count=len(prepared_inputs),
        written_count=len(entries),
        device_used=runtime_label,
        detector_record_count=len(detector_records),
        correction_record_count=len(correction_records),
        correction_applied_count=sum(
            1 for record in correction_records if record.merged_changed
        ),
    )


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


def _open_ocr_backend(
    config: OCRBackendConfig,
    *,
    allow_hf_auto_cpu_fallback: bool,
    verbose: bool,
) -> tuple[Any, str]:
    if config is None:
        raise RuntimeError("OCR backend config is required")

    runtime_used = config.device or config.profile
    try:
        return _attempt_open_ocr_backend(config, verbose=verbose), runtime_used
    except Exception as exc:
        if allow_hf_auto_cpu_fallback and config.engine is OCREngine.HF and config.device == "gpu":
            if verbose:
                logger.warning("GPU init failed (%s). Retrying with CPU.", exc)
            cpu_config = replace(config, device="cpu")
            return _attempt_open_ocr_backend(cpu_config, verbose=verbose), "cpu"
        raise


def _attempt_open_ocr_backend(
    config: OCRBackendConfig,
    *,
    verbose: bool,
) -> Any:
    if verbose:
        logger.info(
            "loading OCR backend: engine=%s role=%s model=%s runtime=%s",
            config.engine,
            config.role,
            config.model_id,
            config.device or config.profile,
        )
    return create_ocr_backend(config)


def _close_ocr_backend(backend: Any | None, *, verbose: bool) -> None:
    if backend is None:
        return
    if hasattr(backend, "close"):
        try:
            backend.close()
            if verbose:
                logger.info("OCR backend released from device memory")
        except Exception as exc:
            logger.warning("failed to release OCR backend cleanly: %s", exc)


@contextmanager
def _managed_ocr_backend(
    config: OCRBackendConfig,
    *,
    allow_hf_auto_cpu_fallback: bool,
    verbose: bool,
) -> Iterator[tuple[Any, str]]:
    backend = None
    runtime_used = config.device or config.profile
    try:
        backend, runtime_used = _open_ocr_backend(
            config,
            allow_hf_auto_cpu_fallback=allow_hf_auto_cpu_fallback,
            verbose=verbose,
        )
        yield backend, runtime_used
    finally:
        _close_ocr_backend(backend, verbose=verbose)


def _recognize_prepared_inputs(
    prepared_inputs: list[_PreparedOCRInput],
    *,
    backend: Any,
    verbose: bool,
    branch_label: str,
) -> list[str]:
    if not prepared_inputs:
        return []

    recognized: list[str] = []
    total = len(prepared_inputs)
    for processed, item in enumerate(prepared_inputs, start=1):
        if verbose:
            logger.info(
                "OCR started: branch=%s %s",
                branch_label,
                _progress_label(processed, total),
            )
        ocr_started = time.monotonic()
        text = _recognize_single_image(backend, item.image)
        recognized.append(text)
        if verbose:
            state = "accepted" if text else "skipped (empty)"
            logger.info(
                "OCR finished: branch=%s %s %s elapsed=%.2fs",
                branch_label,
                _progress_label(processed, total),
                state,
                time.monotonic() - ocr_started,
            )

    return recognized


def _recognize_single_image(backend: Any, image: Image.Image) -> str:
    if hasattr(backend, "recognize"):
        return backend.recognize(image)
    if hasattr(backend, "recognize_batch"):
        texts = backend.recognize_batch([image])
        if len(texts) != 1:
            raise RuntimeError(
                "OCR backend returned mismatched result count for single-image recognition: "
                f"expected=1 actual={len(texts)}"
            )
        return texts[0]
    raise RuntimeError("OCR backend does not provide recognize() or recognize_batch().")


def _build_hybrid_detector_records(
    prepared_inputs: list[_PreparedOCRInput],
    baseline_texts: list[str],
    *,
    detector_backend_config: OCRBackendConfig,
    verbose: bool,
) -> list[HybridDetectorRecord]:
    wide_inputs = [item for item in prepared_inputs if not _is_tall_subtitle_image(item.image)]
    tall_inputs = [item for item in prepared_inputs if _is_tall_subtitle_image(item.image)]

    detector_specs: list[tuple[str, str, list[_PreparedOCRInput], OCRBackendConfig]] = []
    if wide_inputs:
        detector_specs.append(
            (
                "alternate_read_non_tall",
                "ocr-fast",
                wide_inputs,
                replace(detector_backend_config, role="ocr-fast"),
            )
        )
    if tall_inputs:
        detector_specs.append(
            (
                "repeat_drift_tall",
                "detector",
                tall_inputs,
                replace(detector_backend_config, role="detector"),
            )
        )

    detector_records: list[HybridDetectorRecord] = []
    baseline_by_index = {
        item.index: text for item, text in zip(prepared_inputs, baseline_texts, strict=True)
    }

    for branch_name, option_role, branch_inputs, config in detector_specs:
        with _managed_ocr_backend(
            config,
            allow_hf_auto_cpu_fallback=False,
            verbose=verbose,
        ) as (backend, _):
            option_texts = _recognize_prepared_inputs(
                branch_inputs,
                backend=backend,
                verbose=verbose,
                branch_label=f"detector-{branch_name}",
            )
            for item, option_text in zip(branch_inputs, option_texts, strict=True):
                baseline_text = baseline_by_index[item.index]
                if baseline_text == option_text:
                    continue
                ratio = float(item.image.height) / float(item.image.width) if item.image.width > 0 else 0.0
                diff = assess_difference(baseline_text, option_text)
                detector_records.append(
                    HybridDetectorRecord(
                        index=item.index,
                        raw_index=item.frame.raw_index,
                        window_id=item.frame.window_id,
                        start_ms=_timedelta_to_ms(item.frame.start),
                        end_ms=_timedelta_to_ms(item.frame.end),
                        detector_branch=branch_name,
                        shape="tall" if branch_name == "repeat_drift_tall" else "wide",
                        ratio=ratio,
                        option_role=option_role,
                        baseline_text=baseline_text,
                        option_text=option_text,
                        diff_label=diff.label,
                        meaningful=diff.meaningful,
                        char_error_rate=diff.char_error_rate,
                    )
                )
    return detector_records


def _apply_conservative_corrections(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    detector_records: list[HybridDetectorRecord],
    corrector_config: CorrectorConfig,
    runtime_profile: str,
    runtime_binary_path: Path | None,
    runtime_host: str,
    verbose: bool,
) -> list[ConservativeCorrectionRecord]:
    if not detector_records:
        return []

    if corrector_config.mode is CorrectorMode.QWEN_LOCAL:
        return _apply_local_qwen_corrections(
            prepared_inputs=prepared_inputs,
            detector_records=detector_records,
            corrector_config=corrector_config,
            runtime_profile=runtime_profile,
            runtime_binary_path=runtime_binary_path,
            runtime_host=runtime_host,
            verbose=verbose,
        )
    if corrector_config.mode is CorrectorMode.GEMINI:
        return _apply_gemini_corrections(
            prepared_inputs=prepared_inputs,
            detector_records=detector_records,
            corrector_config=corrector_config,
            verbose=verbose,
        )
    raise AssertionError(f"unhandled corrector mode: {corrector_config.mode}")


def _apply_local_qwen_corrections(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    detector_records: list[HybridDetectorRecord],
    corrector_config: CorrectorConfig,
    runtime_profile: str,
    runtime_binary_path: Path | None,
    runtime_host: str,
    verbose: bool,
) -> list[ConservativeCorrectionRecord]:
    if corrector_config.local_model_path is None or corrector_config.local_mmproj_path is None:
        raise RuntimeError("qwen-local correction requires explicit corrector model and mmproj paths")

    correction_inputs = [prepared_inputs[record.index] for record in detector_records]
    corrector_backend_config = OCRBackendConfig(
        engine=OCREngine.LLAMA_SERVER,
        model_id=str(corrector_config.local_model_path),
        model_path=corrector_config.local_model_path,
        mmproj_path=corrector_config.local_mmproj_path,
        max_new_tokens=LOCAL_QWEN_MAX_NEW_TOKENS,
        local_files_only=True,
        role="corrector",
        prompt_text=STRICT_OCR_V1_PROMPT,
        profile=runtime_profile,
        binary_path=runtime_binary_path,
        host=runtime_host,
        port=corrector_config.port,
        ctx_size=LOCAL_QWEN_CTX_SIZE,
        n_predict=LOCAL_QWEN_MAX_NEW_TOKENS,
        reasoning="off",
        no_mmproj_offload=corrector_config.local_no_mmproj_offload,
        startup_timeout_sec=corrector_config.startup_timeout_sec,
    )
    with _managed_ocr_backend(
        corrector_backend_config,
        allow_hf_auto_cpu_fallback=False,
        verbose=verbose,
    ) as (corrector_backend, _):
        corrector_texts = _recognize_prepared_inputs(
            correction_inputs,
            backend=corrector_backend,
            verbose=verbose,
            branch_label="corrector-qwen-local",
        )

    corrector_name = corrector_name_for_config(corrector_config)
    records: list[ConservativeCorrectionRecord] = []
    for detector_record, corrector_text in zip(detector_records, corrector_texts, strict=True):
        prompt_style = corrector_prompt_for_shape(corrector_config, detector_record.shape)[1]
        records.append(
            _build_correction_record(
                detector_record=detector_record,
                corrector_name=corrector_name,
                corrector_prompt_style=prompt_style,
                corrector_text=corrector_text,
                corrector_reasoning_content="",
            )
        )
    return records


def _log_runtime_selection(*, engine: OCREngine, runtime_used: str) -> None:
    if engine is OCREngine.HF:
        logger.info("using HF device: %s", runtime_used)
        return
    logger.info("using llama-server runtime profile: %s", runtime_used)


def _apply_gemini_corrections(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    detector_records: list[HybridDetectorRecord],
    corrector_config: CorrectorConfig,
    verbose: bool,
) -> list[ConservativeCorrectionRecord]:
    prepared_by_index = {item.index: item for item in prepared_inputs}
    corrector_name = corrector_name_for_config(corrector_config)
    records: list[ConservativeCorrectionRecord] = []
    for detector_record in detector_records:
        if verbose:
            logger.info(
                "running Gemini corrector: row=%s branch=%s shape=%s",
                detector_record.index,
                detector_record.detector_branch,
                detector_record.shape,
            )
        prepared = prepared_by_index[detector_record.index]
        corrector_text, prompt_style, reasoning_content = request_gemini_correction(
            config=corrector_config,
            image=prepared.image,
            shape=detector_record.shape,
        )
        records.append(
            _build_correction_record(
                detector_record=detector_record,
                corrector_name=corrector_name,
                corrector_prompt_style=prompt_style,
                corrector_text=corrector_text,
                corrector_reasoning_content=reasoning_content,
            )
        )
    return records


def _build_correction_record(
    *,
    detector_record: HybridDetectorRecord,
    corrector_name: str,
    corrector_prompt_style: str,
    corrector_text: str,
    corrector_reasoning_content: str,
) -> ConservativeCorrectionRecord:
    anchor_rows = build_focus_context(detector_record.baseline_text, detector_record.option_text)
    merged = apply_union_anchor_merge(
        detector_record.baseline_text,
        corrector_text,
        anchor_rows,
    )
    return ConservativeCorrectionRecord(
        index=detector_record.index,
        raw_index=detector_record.raw_index,
        window_id=detector_record.window_id,
        start_ms=detector_record.start_ms,
        end_ms=detector_record.end_ms,
        detector_branch=detector_record.detector_branch,
        shape=detector_record.shape,
        ratio=detector_record.ratio,
        option_role=detector_record.option_role,
        baseline_text=detector_record.baseline_text,
        option_text=detector_record.option_text,
        diff_label=detector_record.diff_label,
        meaningful=detector_record.meaningful,
        char_error_rate=detector_record.char_error_rate,
        anchor_count=len(anchor_rows),
        corrector_name=corrector_name,
        corrector_prompt_style=corrector_prompt_style,
        corrector_text=corrector_text,
        conservative_merged_text=merged.merged_text,
        applied_op_count=len(merged.applied_ops),
        raw_changed=corrector_text != detector_record.baseline_text,
        merged_changed=merged.merged_text != detector_record.baseline_text,
        corrector_reasoning_content=corrector_reasoning_content,
    )


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


def _timedelta_to_ms(value: timedelta) -> int:
    return int(round(value.total_seconds() * 1000.0))


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
