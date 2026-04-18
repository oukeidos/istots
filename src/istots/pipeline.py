from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import Counter
import difflib
import multiprocessing
import os
from tempfile import TemporaryDirectory
import logging
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterator, TypeVar

from PIL import Image

from istots.anchor_merge import apply_union_anchor_merge, build_focus_context
from istots.corrector import (
    GeminiRequestFailedError,
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
from istots.llama_runtime import DEFAULT_ROLE_PORTS, LlamaServerRole
from istots.ocr import (
    LOCAL_PADDLE_CTX_SIZE,
    OCRBackendConfig,
    OCREngine,
    PaddleOCRVLRuntimeOverrides,
    Qwen35RuntimeOverrides,
    ResolvedLlamaRuntimeOverrides,
    create_ocr_backend,
    normalize_ocr_engine,
)
from istots.srt_writer import SubtitleEntry, write_srt
from istots.sup_reader import iter_sup_window_frames, release_parser_predecode_workers
from istots.text_diff import DEFAULT_TEXT_DIFF_PROFILE, assess_difference

logger = logging.getLogger(__name__)
TALL_SUBTITLE_RATIO_THRESHOLD = 2.0
HF_FAST_MIN_PIXELS = 32768
KANJI_FAMILY_MIN_COUNT = 2
MAX_FAMILY_AGREEMENT_ROWS_PER_SUPPORT_ROW = 10
_PREPARED_INPUT_SUBPROCESS_ENV = "ISTOTS_PREPARE_OCR_INPUTS_IN_SUBPROCESS"
_PREPARED_INPUT_SPILL_FORMAT_ENV = "ISTOTS_PREPARED_INPUT_SPILL_FORMAT"
_T = TypeVar("_T")


@dataclass
class ConversionResult:
    output_srt: Path
    processed_count: int
    written_count: int
    device_used: str
    detector_record_count: int = 0
    correction_record_count: int = 0
    correction_applied_count: int = 0
    correction_fallback_count: int = 0


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
    raw_index: int
    window_id: int
    left: int
    top: int
    right: int
    bottom: int
    start: timedelta
    end: timedelta
    image_width: int
    image_height: int
    image_mode: str
    image: Image.Image | None
    image_path: Path | None = None


@dataclass(frozen=True)
class _GeminiCorrectionOutcome:
    corrector_text: str
    prompt_style: str
    reasoning_content: str
    status: str
    error_message: str = ""


@dataclass(frozen=True)
class _KanjiFamilyCandidateStats:
    family: str
    support_rows: int
    pure_rows: int
    mixed_rows: int
    agreement_rows: int


def _exact_image_identity_key(image: Image.Image) -> tuple[str, tuple[int, int], bytes]:
    return (image.mode, image.size, image.tobytes())


def _prepared_image(prepared: _PreparedOCRInput) -> Image.Image:
    if prepared.image is not None:
        return prepared.image
    if prepared.image_path is None:
        raise RuntimeError("prepared OCR input is missing both in-memory image and image_path")
    with Image.open(prepared.image_path) as image:
        if image.mode == prepared.image_mode:
            return image.copy()
        return image.convert(prepared.image_mode)


def _prepared_identity_key(prepared: _PreparedOCRInput) -> Any:
    if prepared.image_path is not None:
        return ("disk", prepared.image_mode, prepared.image_width, prepared.image_height, str(prepared.image_path))
    if prepared.image is None:
        raise RuntimeError("prepared OCR input is missing both in-memory image and image_path")
    return ("memory", _exact_image_identity_key(prepared.image))


def _prepared_aspect_ratio(prepared: _PreparedOCRInput) -> float:
    if prepared.image_width <= 0:
        return 0.0
    return float(prepared.image_height) / float(prepared.image_width)


def _prepared_is_tall(prepared: _PreparedOCRInput) -> bool:
    return _prepared_aspect_ratio(prepared) >= TALL_SUBTITLE_RATIO_THRESHOLD


def _prepare_inputs_in_subprocess_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    raw = os.getenv(_PREPARED_INPUT_SUBPROCESS_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _prepared_input_spill_encoding() -> tuple[str, str, dict[str, Any]]:
    raw = os.getenv(_PREPARED_INPUT_SPILL_FORMAT_ENV, "png").strip().lower()
    if raw in {"", "png", "png-default"}:
        return ".png", "PNG", {}
    if raw in {"png-fast", "png0"}:
        return ".png", "PNG", {"compress_level": 0, "optimize": False}
    if raw == "bmp":
        return ".bmp", "BMP", {}
    raise ValueError(
        f"unsupported prepared input spill format: {raw!r}; expected one of 'png', 'png-fast', 'bmp'"
    )


def _spill_prepared_inputs_to_directory(
    prepared_inputs: list[_PreparedOCRInput],
    *,
    output_dir: Path,
) -> list[_PreparedOCRInput]:
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_to_path: dict[tuple[str, tuple[int, int], bytes], Path] = {}
    spilled_inputs: list[_PreparedOCRInput] = []
    suffix, image_format, save_kwargs = _prepared_input_spill_encoding()

    for item in prepared_inputs:
        if item.image is None:
            raise RuntimeError("prepared OCR input already spilled before disk materialization")
        identity = _exact_image_identity_key(item.image)
        image_path = identity_to_path.get(identity)
        if image_path is None:
            image_path = output_dir / f"{len(identity_to_path):06d}{suffix}"
            item.image.save(image_path, format=image_format, **save_kwargs)
            identity_to_path[identity] = image_path
        spilled_inputs.append(replace(item, image=None, image_path=image_path))

    return spilled_inputs


def _prepare_ocr_inputs_worker_entry(
    input_sup: str,
    max_items: int | None,
    enable_furigana_mask: bool,
    spill_dir: str,
    connection: Any,
) -> None:
    try:
        prepared_inputs = _collect_prepared_ocr_inputs_inprocess(
            Path(input_sup),
            max_items=max_items,
            enable_furigana_mask=enable_furigana_mask,
            verbose=False,
        )
        connection.send(
            (
                "ok",
                _spill_prepared_inputs_to_directory(
                    prepared_inputs,
                    output_dir=Path(spill_dir),
                ),
            )
        )
    except Exception as exc:  # pragma: no cover - exercised through parent wrapper
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:  # pragma: no branch
        connection.close()


def _collect_prepared_ocr_inputs_via_subprocess(
    input_sup: Path,
    *,
    max_items: int | None,
    enable_furigana_mask: bool,
    verbose: bool,
    spill_dir: Path,
) -> list[_PreparedOCRInput]:
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_prepare_ocr_inputs_worker_entry,
        args=(
            str(input_sup),
            max_items,
            enable_furigana_mask,
            str(spill_dir),
            child_conn,
        ),
        daemon=False,
    )
    process.start()
    child_conn.close()
    try:
        status, payload = parent_conn.recv()
    finally:
        parent_conn.close()
        process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"prepared-input subprocess failed with exit code {process.exitcode}")
    if status != "ok":
        raise RuntimeError(f"prepared-input subprocess failed: {payload}")
    if verbose:
        logger.info("prepared-input subprocess finished: rows=%d spill_dir=%s", len(payload), spill_dir)
    return payload


def _dedupe_by_exact_image_identity(
    items: list[_T],
    *,
    image_getter: Callable[[_T], Image.Image],
    discriminator_getter: Callable[[_T], Any] | None = None,
) -> tuple[list[_T], list[int], list[int]]:
    key_to_unique_index: dict[Any, int] = {}
    unique_items: list[_T] = []
    item_to_unique_index: list[int] = []
    unique_group_sizes: list[int] = []

    for item in items:
        image = image_getter(item)
        key: Any = _exact_image_identity_key(image)
        if discriminator_getter is not None:
            key = (key, discriminator_getter(item))
        unique_index = key_to_unique_index.get(key)
        if unique_index is None:
            unique_index = len(unique_items)
            key_to_unique_index[key] = unique_index
            unique_items.append(item)
            unique_group_sizes.append(0)
        unique_group_sizes[unique_index] += 1
        item_to_unique_index.append(unique_index)

    return unique_items, item_to_unique_index, unique_group_sizes


def _resolve_paddle_runtime_overrides(
    *,
    role: str,
    overrides: PaddleOCRVLRuntimeOverrides,
) -> ResolvedLlamaRuntimeOverrides:
    normalized_role = LlamaServerRole(role)
    return ResolvedLlamaRuntimeOverrides(
        profile=overrides.profile,
        port=overrides.port if overrides.port is not None else DEFAULT_ROLE_PORTS[normalized_role],
        threads=overrides.threads,
        threads_batch=overrides.threads_batch,
        ctx_size=overrides.ctx_size if overrides.ctx_size is not None else LOCAL_PADDLE_CTX_SIZE,
        gpu_layers=overrides.gpu_layers,
        no_mmproj_offload=overrides.no_mmproj_offload,
        startup_timeout_sec=overrides.startup_timeout_sec,
    )


def _resolve_qwen_runtime_overrides(
    overrides: Qwen35RuntimeOverrides,
) -> ResolvedLlamaRuntimeOverrides:
    return ResolvedLlamaRuntimeOverrides(
        profile=overrides.profile,
        port=overrides.port if overrides.port is not None else DEFAULT_ROLE_PORTS[LlamaServerRole.CORRECTOR],
        threads=overrides.threads,
        threads_batch=overrides.threads_batch,
        gpu_layers=overrides.gpu_layers,
        no_mmproj_offload=overrides.no_mmproj_offload,
        startup_timeout_sec=overrides.startup_timeout_sec,
        ctx_size=overrides.ctx_size if overrides.ctx_size is not None else LOCAL_QWEN_CTX_SIZE,
        n_predict=overrides.n_predict if overrides.n_predict is not None else LOCAL_QWEN_MAX_NEW_TOKENS,
        reasoning=overrides.reasoning if overrides.reasoning is not None else "off",
    )


def _build_paddle_backend_config(
    *,
    base_config: OCRBackendConfig,
    role: str,
    overrides: PaddleOCRVLRuntimeOverrides,
) -> OCRBackendConfig:
    resolved = _resolve_paddle_runtime_overrides(role=role, overrides=overrides)
    return replace(
        base_config,
        role=role,
        profile=resolved.profile,
        port=resolved.port,
        threads=resolved.threads,
        threads_batch=resolved.threads_batch,
        ctx_size=resolved.ctx_size,
        gpu_layers=resolved.gpu_layers,
        no_mmproj_offload=resolved.no_mmproj_offload,
        startup_timeout_sec=resolved.startup_timeout_sec,
    )


def _build_fast_backend_config(
    *,
    base_config: OCRBackendConfig,
    role: str,
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides,
) -> OCRBackendConfig:
    if base_config.engine is OCREngine.LLAMA_SERVER:
        return _build_paddle_backend_config(
            base_config=base_config,
            role=role,
            overrides=paddle_runtime_overrides,
        )
    if role == "ocr-fast":
        return replace(base_config, role=role, hf_min_pixels=HF_FAST_MIN_PIXELS)
    return replace(base_config, role=role)


def convert_sup_to_srt(
    input_sup: Path,
    output_srt: Path,
    hf_device: str = "auto",
    hf_dtype: str = "auto",
    engine: str | OCREngine = OCREngine.HF,
    ocr_mode: str = "default",
    detector_output: Path | None = None,
    detector_mode: str = "default",
    corrector_config: CorrectorConfig | None = None,
    model_id: str = "PaddlePaddle/PaddleOCR-VL-1.5",
    models_dir: Path | None = None,
    max_items: int | None = None,
    max_new_tokens: int = 256,
    local_files_only: bool = True,
    enable_furigana_mask: bool = False,
    detector_family_addon: bool = False,
    srt_policy: str = "safe",
    runtime_binary_path: Path | None = None,
    runtime_host: str = "127.0.0.1",
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides | None = None,
    use_temp_ocr_image_files: bool | None = None,
    verbose: bool = True,
) -> ConversionResult:
    if not input_sup.exists():
        raise FileNotFoundError(f"Input SUP file not found: {input_sup}")

    logger.info("starting conversion: input=%s output=%s", input_sup, output_srt)
    normalized_engine = normalize_ocr_engine(engine)
    normalized_ocr_mode = _normalize_ocr_mode(ocr_mode)
    normalized_detector_mode = _normalize_detector_mode(detector_mode)
    allow_hf_auto_cpu_fallback = normalized_engine is OCREngine.HF and hf_device == "auto"
    resolved_hf_device = resolve_hf_device(hf_device) if normalized_engine is OCREngine.HF else None
    resolved_paddle_runtime = paddle_runtime_overrides or PaddleOCRVLRuntimeOverrides()
    runtime_label = resolved_hf_device if resolved_hf_device is not None else resolved_paddle_runtime.profile
    if detector_output is not None and normalized_engine is not OCREngine.LLAMA_SERVER:
        raise ValueError("detector output requires the llama-server engine")
    if detector_output is not None and normalized_ocr_mode != "default":
        raise ValueError("detector output requires the default OCR mode")
    if normalized_detector_mode != "default" and normalized_engine is not OCREngine.LLAMA_SERVER:
        raise ValueError("detector mode overrides require the llama-server engine")
    if normalized_detector_mode != "default" and normalized_ocr_mode != "default":
        raise ValueError("detector mode overrides require the default OCR mode")
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
        binary_path=runtime_binary_path,
        host=runtime_host,
    )

    if verbose:
        logger.info("ocr mode: %s", normalized_ocr_mode)
        if detector_output is not None or corrector_config is not None:
            logger.info("detector mode: %s", normalized_detector_mode)
        logger.info("furigana masking: %s", "enabled" if enable_furigana_mask else "disabled")
        logger.info(
            "temp OCR image files: %s",
            "enabled" if _prepare_inputs_in_subprocess_enabled(use_temp_ocr_image_files) else "disabled",
        )
        logger.info("srt policy: %s", srt_policy)

    if normalized_ocr_mode == "fast":
        return _convert_sup_to_srt_fast(
            input_sup=input_sup,
            output_srt=output_srt,
            backend_config=backend_config,
            paddle_runtime_overrides=resolved_paddle_runtime,
            runtime_label=runtime_label,
            allow_hf_auto_cpu_fallback=allow_hf_auto_cpu_fallback,
            max_items=max_items,
            enable_furigana_mask=enable_furigana_mask,
            use_temp_ocr_image_files=use_temp_ocr_image_files,
            srt_policy=srt_policy,
            verbose=verbose,
        )
    if detector_output is not None or corrector_config is not None:
        return _convert_sup_to_srt_default_with_detector(
            input_sup=input_sup,
            output_srt=output_srt,
            backend_config=backend_config,
            paddle_runtime_overrides=resolved_paddle_runtime,
            runtime_label=runtime_label,
            detector_output=detector_output,
            detector_mode=normalized_detector_mode,
            corrector_config=corrector_config,
            detector_family_addon=detector_family_addon,
            max_items=max_items,
            enable_furigana_mask=enable_furigana_mask,
            use_temp_ocr_image_files=use_temp_ocr_image_files,
            srt_policy=srt_policy,
            verbose=verbose,
        )

    with _managed_prepared_ocr_inputs(
        input_sup,
        max_items=max_items,
        enable_furigana_mask=enable_furigana_mask,
        use_temp_ocr_image_files=use_temp_ocr_image_files,
        verbose=verbose,
    ) as prepared_inputs:
        with _managed_ocr_backend(
            _build_paddle_backend_config(
                base_config=backend_config,
                role="ocr",
                overrides=resolved_paddle_runtime,
            ) if normalized_engine is OCREngine.LLAMA_SERVER else backend_config,
            allow_hf_auto_cpu_fallback=allow_hf_auto_cpu_fallback,
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
                if (segment := _build_window_text_segment(item, text)) is not None
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
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides,
    runtime_label: str,
    allow_hf_auto_cpu_fallback: bool,
    max_items: int | None,
    enable_furigana_mask: bool,
    use_temp_ocr_image_files: bool | None,
    srt_policy: str,
    verbose: bool,
) -> ConversionResult:
    with _managed_prepared_ocr_inputs(
        input_sup,
        max_items=max_items,
        enable_furigana_mask=enable_furigana_mask,
        use_temp_ocr_image_files=use_temp_ocr_image_files,
        verbose=verbose,
    ) as prepared_inputs:
        wide_inputs = [item for item in prepared_inputs if not _prepared_is_tall(item)]
        tall_inputs = [item for item in prepared_inputs if _prepared_is_tall(item)]

        if verbose:
            logger.info(
                "hybrid OCR partitioned rows: non_tall=%d tall=%d threshold=%.1f",
                len(wide_inputs),
                len(tall_inputs),
                TALL_SUBTITLE_RATIO_THRESHOLD,
            )

        backend_specs: list[tuple[str, OCRBackendConfig]] = []
        if wide_inputs:
            backend_specs.append(
                (
                    "ocr-fast",
                    _build_fast_backend_config(
                        base_config=backend_config,
                        role="ocr-fast",
                        paddle_runtime_overrides=paddle_runtime_overrides,
                    ),
                )
            )
        if tall_inputs:
            backend_specs.append(
                (
                    "ocr",
                    _build_fast_backend_config(
                        base_config=backend_config,
                        role="ocr",
                        paddle_runtime_overrides=paddle_runtime_overrides,
                    ),
                )
            )

        device_logged = False
        active_runtime_label = runtime_label
        active_hf_device = backend_config.device
        recognized_by_index: dict[int, str] = {}

        for role, config in backend_specs:
            branch_inputs = wide_inputs if role == "ocr-fast" else tall_inputs
            branch_label = "non-tall-fast" if role == "ocr-fast" else "tall-default"
            if not branch_inputs:
                continue
            active_config = (
                replace(config, device=active_hf_device)
                if backend_config.engine is OCREngine.HF
                else config
            )
            with _managed_ocr_backend(
                active_config,
                allow_hf_auto_cpu_fallback=allow_hf_auto_cpu_fallback,
                verbose=verbose,
            ) as (backend, runtime_used):
                if backend_config.engine is OCREngine.HF:
                    active_hf_device = runtime_used
                    active_runtime_label = runtime_used
                if verbose and not device_logged:
                    _log_runtime_selection(engine=backend_config.engine, runtime_used=runtime_used)
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
            if (segment := _build_window_text_segment(item, recognized_by_index.get(item.index, ""))) is not None
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
            device_used=active_runtime_label,
        )


def _convert_sup_to_srt_default_with_detector(
    *,
    input_sup: Path,
    output_srt: Path,
    backend_config: OCRBackendConfig,
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides,
    runtime_label: str,
    detector_output: Path | None,
    detector_mode: str,
    corrector_config: CorrectorConfig | None,
    detector_family_addon: bool,
    max_items: int | None,
    enable_furigana_mask: bool,
    use_temp_ocr_image_files: bool | None,
    srt_policy: str,
    verbose: bool,
) -> ConversionResult:
    with _managed_prepared_ocr_inputs(
        input_sup,
        max_items=max_items,
        enable_furigana_mask=enable_furigana_mask,
        use_temp_ocr_image_files=use_temp_ocr_image_files,
        verbose=verbose,
    ) as prepared_inputs:
        with _managed_ocr_backend(
            _build_paddle_backend_config(
                base_config=backend_config,
                role="ocr",
                overrides=paddle_runtime_overrides,
            ),
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

        detector_records = _build_detector_surface_records(
            prepared_inputs=prepared_inputs,
            baseline_texts=baseline_texts,
            detector_backend_config=backend_config,
            paddle_runtime_overrides=paddle_runtime_overrides,
            detector_mode=detector_mode,
            verbose=verbose,
        )
        if detector_family_addon:
            dominant_family, addon_records = _build_dominant_family_addon_records(
                prepared_inputs=prepared_inputs,
                baseline_texts=baseline_texts,
                s1_detector_records=detector_records,
            )
            detector_records.extend(addon_records)
            if verbose:
                if dominant_family is None:
                    logger.info("dominant-family detector add-on: no repeated single-char kanji family found")
                else:
                    logger.info(
                        "dominant-family detector add-on: family=%s added_rows=%d",
                        dominant_family,
                        len(addon_records),
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
            if (segment := _build_window_text_segment(item, text)) is not None
        ]
        entries = _build_subtitle_entries(segments, srt_policy=srt_policy)
        write_srt(entries, output_srt)
        correction_fallback_count = sum(
            1 for record in correction_records if record.corrector_status == "fallback_baseline"
        )
        if verbose:
            logger.info(
                "conversion finished: processed=%d written=%d detector_disagreements=%d correction_rows=%d correction_fallbacks=%d output=%s",
                len(prepared_inputs),
                len(entries),
                len(detector_records),
                len(correction_records),
                correction_fallback_count,
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
            correction_fallback_count=correction_fallback_count,
        )


def _normalize_ocr_mode(ocr_mode: str) -> str:
    normalized = ocr_mode.strip().lower()
    if normalized in {"default", "fast"}:
        return normalized
    raise ValueError(f"unsupported OCR mode: {ocr_mode!r}")


def _normalize_detector_mode(detector_mode: str) -> str:
    normalized = detector_mode.strip().lower()
    if normalized in {"default", "wider"}:
        return normalized
    raise ValueError(f"unsupported detector mode: {detector_mode!r}")


@contextmanager
def _managed_prepared_ocr_inputs(
    input_sup: Path,
    *,
    max_items: int | None,
    enable_furigana_mask: bool,
    use_temp_ocr_image_files: bool | None,
    verbose: bool,
) -> Iterator[list[_PreparedOCRInput]]:
    if not _prepare_inputs_in_subprocess_enabled(use_temp_ocr_image_files):
        yield _collect_prepared_ocr_inputs_inprocess(
            input_sup,
            max_items=max_items,
            enable_furigana_mask=enable_furigana_mask,
            verbose=verbose,
        )
        return

    with TemporaryDirectory(prefix="istots-prepared-inputs-") as temp_dir:
        spill_dir = Path(temp_dir)
        if verbose:
            logger.info("collecting prepared OCR inputs in subprocess: spill_dir=%s", spill_dir)
        yield _collect_prepared_ocr_inputs_via_subprocess(
            input_sup,
            max_items=max_items,
            enable_furigana_mask=enable_furigana_mask,
            verbose=verbose,
            spill_dir=spill_dir,
        )


def _collect_prepared_ocr_inputs(
    input_sup: Path,
    *,
    max_items: int | None,
    enable_furigana_mask: bool,
    verbose: bool,
) -> list[_PreparedOCRInput]:
    return _collect_prepared_ocr_inputs_inprocess(
        input_sup,
        max_items=max_items,
        enable_furigana_mask=enable_furigana_mask,
        verbose=verbose,
    )


def _collect_prepared_ocr_inputs_inprocess(
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

    try:
        frames = list(iter_sup_window_frames(input_sup, max_items=max_items, on_total=on_total))
    finally:
        release_parser_predecode_workers()
    if enable_furigana_mask:
        if verbose:
            logger.info("building furigana mask statistics by orientation")
        images = [result.image for result in build_furigana_masks([frame.image for frame in frames])]
    else:
        images = [frame.image for frame in frames]
    return [
        _PreparedOCRInput(
            index=index,
            raw_index=frame.raw_index,
            window_id=frame.window_id,
            left=frame.left,
            top=frame.top,
            right=frame.right,
            bottom=frame.bottom,
            start=frame.start,
            end=frame.end,
            image_width=image.width,
            image_height=image.height,
            image_mode=image.mode,
            image=image,
        )
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

    key_to_unique_index: dict[Any, int] = {}
    unique_inputs: list[_PreparedOCRInput] = []
    item_to_unique_index: list[int] = []
    unique_group_sizes: list[int] = []

    for item in prepared_inputs:
        key = _prepared_identity_key(item)
        unique_index = key_to_unique_index.get(key)
        if unique_index is None:
            unique_index = len(unique_inputs)
            key_to_unique_index[key] = unique_index
            unique_inputs.append(item)
            unique_group_sizes.append(0)
        unique_group_sizes[unique_index] += 1
        item_to_unique_index.append(unique_index)

    recognized_unique: list[str] = []
    total = len(unique_inputs)
    for processed, item in enumerate(unique_inputs, start=1):
        unique_index = processed - 1
        if verbose:
            logger.info(
                "OCR started: branch=%s %s rows=%d",
                branch_label,
                _progress_label(processed, total),
                unique_group_sizes[unique_index],
            )
        ocr_started = time.monotonic()
        text = _recognize_single_image(backend, _prepared_image(item))
        recognized_unique.append(text)
        if verbose:
            state = "accepted" if text else "skipped (empty)"
            logger.info(
                "OCR finished: branch=%s %s %s rows=%d elapsed=%.2fs",
                branch_label,
                _progress_label(processed, total),
                state,
                unique_group_sizes[unique_index],
                time.monotonic() - ocr_started,
            )

    return [recognized_unique[unique_index] for unique_index in item_to_unique_index]


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
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides,
    verbose: bool,
) -> list[HybridDetectorRecord]:
    wide_inputs = [item for item in prepared_inputs if not _prepared_is_tall(item)]
    tall_inputs = [item for item in prepared_inputs if _prepared_is_tall(item)]

    detector_specs: list[tuple[str, str, list[_PreparedOCRInput], OCRBackendConfig]] = []
    if wide_inputs:
        detector_specs.append(
            (
                "alternate_read_non_tall",
                "ocr-fast",
                wide_inputs,
                _build_paddle_backend_config(
                    base_config=detector_backend_config,
                    role="ocr-fast",
                    overrides=paddle_runtime_overrides,
                ),
            )
        )
    if tall_inputs:
        detector_specs.append(
            (
                "repeat_drift_tall",
                "detector",
                tall_inputs,
                _build_paddle_backend_config(
                    base_config=detector_backend_config,
                    role="detector",
                    overrides=paddle_runtime_overrides,
                ),
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
                ratio = _prepared_aspect_ratio(item)
                diff = assess_difference(
                    baseline_text,
                    option_text,
                    profile=DEFAULT_TEXT_DIFF_PROFILE,
                )
                detector_records.append(
                    HybridDetectorRecord(
                        index=item.index,
                        raw_index=item.raw_index,
                        window_id=item.window_id,
                        start_ms=_timedelta_to_ms(item.start),
                        end_ms=_timedelta_to_ms(item.end),
                        detector_branch=branch_name,
                        shape="tall" if branch_name == "repeat_drift_tall" else "wide",
                        ratio=ratio,
                        option_role=option_role,
                        baseline_text=baseline_text,
                        option_text=option_text,
                        diff_label=diff.label,
                        meaningful=diff.meaningful,
                        char_error_rate=diff.char_error_rate,
                        source_tags=("hybrid_detector",),
                        alternate_source_kind=(
                            "min32768" if branch_name == "alternate_read_non_tall" else "temp0_repeat"
                        ),
                    )
                )
    return detector_records


def _build_detector_surface_records(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    baseline_texts: list[str],
    detector_backend_config: OCRBackendConfig,
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides,
    detector_mode: str,
    verbose: bool,
) -> list[HybridDetectorRecord]:
    detector_records = _build_hybrid_detector_records(
        prepared_inputs,
        baseline_texts,
        detector_backend_config=detector_backend_config,
        paddle_runtime_overrides=paddle_runtime_overrides,
        verbose=verbose,
    )
    if detector_mode != "wider":
        return detector_records

    p2_records = _build_p2_meaningful_temp0_records(
        prepared_inputs=prepared_inputs,
        baseline_texts=baseline_texts,
        detector_backend_config=detector_backend_config,
        paddle_runtime_overrides=paddle_runtime_overrides,
        verbose=verbose,
    )
    merged_records = _merge_detector_surface_records(
        primary_records=detector_records,
        extra_records=p2_records,
    )
    if verbose:
        overlap_count = len(detector_records) + len(p2_records) - len(merged_records)
        logger.info(
            "wider detector surface: s1_rows=%d p2_rows=%d overlap_rows=%d total_rows=%d",
            len(detector_records),
            len(p2_records),
            overlap_count,
            len(merged_records),
        )
    return merged_records


def _build_p2_meaningful_temp0_records(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    baseline_texts: list[str],
    detector_backend_config: OCRBackendConfig,
    paddle_runtime_overrides: PaddleOCRVLRuntimeOverrides,
    verbose: bool,
) -> list[HybridDetectorRecord]:
    if not prepared_inputs:
        return []

    baseline_by_index = dict(zip((item.index for item in prepared_inputs), baseline_texts, strict=True))
    detector_records: list[HybridDetectorRecord] = []
    config = _build_paddle_backend_config(
        base_config=detector_backend_config,
        role="detector",
        overrides=paddle_runtime_overrides,
    )
    with _managed_ocr_backend(
        config,
        allow_hf_auto_cpu_fallback=False,
        verbose=verbose,
    ) as (backend, _):
        option_texts = _recognize_prepared_inputs(
            prepared_inputs,
            backend=backend,
            verbose=verbose,
            branch_label="detector-p2_meaningful_temp0",
        )
        for item, option_text in zip(prepared_inputs, option_texts, strict=True):
            baseline_text = baseline_by_index[item.index]
            if baseline_text == option_text:
                continue
            ratio = _prepared_aspect_ratio(item)
            diff = assess_difference(
                baseline_text,
                option_text,
                profile=DEFAULT_TEXT_DIFF_PROFILE,
            )
            if not diff.meaningful:
                continue
            detector_records.append(
                HybridDetectorRecord(
                    index=item.index,
                    raw_index=item.raw_index,
                    window_id=item.window_id,
                    start_ms=_timedelta_to_ms(item.start),
                    end_ms=_timedelta_to_ms(item.end),
                    detector_branch="p2_meaningful_temp0",
                    shape="tall" if _prepared_is_tall(item) else "wide",
                    ratio=ratio,
                    option_role="detector",
                    baseline_text=baseline_text,
                    option_text=option_text,
                    diff_label=diff.label,
                    meaningful=diff.meaningful,
                    char_error_rate=diff.char_error_rate,
                    source_tags=("p2_meaningful_temp0",),
                    alternate_source_kind="temp0_repeat",
                )
            )
    return detector_records


def _merge_detector_surface_records(
    *,
    primary_records: list[HybridDetectorRecord],
    extra_records: list[HybridDetectorRecord],
) -> list[HybridDetectorRecord]:
    merged_records = list(primary_records)
    record_index = {record.index: idx for idx, record in enumerate(merged_records)}
    for record in extra_records:
        existing_idx = record_index.get(record.index)
        if existing_idx is None:
            record_index[record.index] = len(merged_records)
            merged_records.append(record)
            continue
        existing = merged_records[existing_idx]
        merged_records[existing_idx] = replace(
            existing,
            source_tags=_merge_source_tags(existing.source_tags, record.source_tags),
        )
    return merged_records


def _merge_source_tags(*tag_groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for tags in tag_groups:
        for tag in tags:
            if tag not in merged:
                merged.append(tag)
    return tuple(merged)


def _build_dominant_family_addon_records(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    baseline_texts: list[str],
    s1_detector_records: list[HybridDetectorRecord],
) -> tuple[str | None, list[HybridDetectorRecord]]:
    dominant_family_stats = _select_dominant_kanji_family(
        prepared_inputs=prepared_inputs,
        baseline_texts=baseline_texts,
        s1_detector_records=s1_detector_records,
    )
    if dominant_family_stats is None:
        return None, []
    dominant_family = dominant_family_stats.family

    disagreement_indices = {record.index for record in s1_detector_records}
    addon_records: list[HybridDetectorRecord] = []
    for item, baseline_text in zip(prepared_inputs, baseline_texts, strict=True):
        if item.index in disagreement_indices or not baseline_text:
            continue
        family_swap = _build_family_pair_swap_text(baseline_text, dominant_family)
        if family_swap is None:
            continue
        option_text, current_char, alternate_char = family_swap
        ratio = _prepared_aspect_ratio(item)
        diff = assess_difference(
            baseline_text,
            option_text,
            profile=DEFAULT_TEXT_DIFF_PROFILE,
        )
        addon_records.append(
            HybridDetectorRecord(
                index=item.index,
                raw_index=item.raw_index,
                window_id=item.window_id,
                start_ms=_timedelta_to_ms(item.start),
                end_ms=_timedelta_to_ms(item.end),
                detector_branch="dominant_family_addon",
                shape="tall" if _prepared_is_tall(item) else "wide",
                ratio=ratio,
                option_role="dominant-family-addon",
                baseline_text=baseline_text,
                option_text=option_text,
                diff_label=diff.label,
                meaningful=diff.meaningful,
                char_error_rate=diff.char_error_rate,
                source_tags=("dominant_family_addon",),
                alternate_source_kind="family_pair_swap",
                dominant_family=dominant_family,
                family_current_char=current_char,
                family_alternate_char=alternate_char,
                family_support_rows=dominant_family_stats.support_rows,
                family_pure_rows=dominant_family_stats.pure_rows,
                family_mixed_rows=dominant_family_stats.mixed_rows,
                family_agreement_rows=dominant_family_stats.agreement_rows,
            )
        )
    return dominant_family, addon_records


def _select_dominant_kanji_family(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    baseline_texts: list[str],
    s1_detector_records: list[HybridDetectorRecord],
) -> _KanjiFamilyCandidateStats | None:
    candidates = _collect_kanji_family_candidate_stats(
        prepared_inputs=prepared_inputs,
        baseline_texts=baseline_texts,
        s1_detector_records=s1_detector_records,
    )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.support_rows >= KANJI_FAMILY_MIN_COUNT
        and candidate.pure_rows >= 1
        and candidate.agreement_rows >= 1
        and candidate.agreement_rows
        <= candidate.support_rows * MAX_FAMILY_AGREEMENT_ROWS_PER_SUPPORT_ROW
    ]
    if not eligible:
        return None

    eligible.sort(
        key=lambda candidate: (
            -candidate.support_rows,
            -candidate.pure_rows,
            candidate.mixed_rows,
            candidate.agreement_rows,
            candidate.family,
        )
    )
    best = eligible[0]
    if len(eligible) == 1:
        return best
    second = eligible[1]
    if _family_selection_rank(best) == _family_selection_rank(second):
        return None
    return best


def _family_selection_rank(candidate: _KanjiFamilyCandidateStats) -> tuple[int, int, int, int]:
    return (
        candidate.support_rows,
        candidate.pure_rows,
        -candidate.mixed_rows,
        -candidate.agreement_rows,
    )


def _collect_kanji_family_candidate_stats(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    baseline_texts: list[str],
    s1_detector_records: list[HybridDetectorRecord],
) -> list[_KanjiFamilyCandidateStats]:
    support_rows: Counter[str] = Counter()
    pure_rows: Counter[str] = Counter()
    mixed_rows: Counter[str] = Counter()

    for record in s1_detector_records:
        families = sorted(
            set(
                _iter_single_char_kanji_replace_families(
                    record.baseline_text,
                    record.option_text,
                )
            )
        )
        if not families:
            continue
        for family in families:
            support_rows[family] += 1
        if len(families) == 1:
            pure_rows[families[0]] += 1
        else:
            for family in families:
                mixed_rows[family] += 1

    if not support_rows:
        return []

    disagreement_indices = {record.index for record in s1_detector_records}
    agreement_rows: Counter[str] = Counter()
    candidate_families = tuple(sorted(support_rows))
    for item, baseline_text in zip(prepared_inputs, baseline_texts, strict=True):
        if item.index in disagreement_indices or not baseline_text:
            continue
        for family in candidate_families:
            if _build_family_pair_swap_text(baseline_text, family) is not None:
                agreement_rows[family] += 1

    return [
        _KanjiFamilyCandidateStats(
            family=family,
            support_rows=support_rows[family],
            pure_rows=pure_rows[family],
            mixed_rows=mixed_rows[family],
            agreement_rows=agreement_rows[family],
        )
        for family in candidate_families
    ]


def _iter_single_char_kanji_replace_families(
    baseline_text: str,
    option_text: str,
) -> Iterator[str]:
    matcher = difflib.SequenceMatcher(a=baseline_text, b=option_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        left = baseline_text[i1:i2]
        right = option_text[j1:j2]
        if len(left) != 1 or len(right) != 1 or left == right:
            continue
        if not (_is_kanji_char(left) and _is_kanji_char(right)):
            continue
        yield "".join(sorted(left + right))


def _build_family_pair_swap_text(
    baseline_text: str,
    family: str,
) -> tuple[str, str, str] | None:
    family_chars = tuple(family)
    if len(family_chars) != 2:
        return None
    hits = [char for char in baseline_text if char in family_chars]
    if not hits:
        return None
    unique_hits = set(hits)
    if len(unique_hits) != 1:
        return None
    current_char = hits[0]
    alternate_char = family_chars[1] if current_char == family_chars[0] else family_chars[0]
    swapped = baseline_text.replace(current_char, alternate_char)
    if swapped == baseline_text:
        return None
    return swapped, current_char, alternate_char


def _is_kanji_char(text: str) -> bool:
    if len(text) != 1:
        return False
    codepoint = ord(text)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x3134F
    )


def _apply_conservative_corrections(
    *,
    prepared_inputs: list[_PreparedOCRInput],
    detector_records: list[HybridDetectorRecord],
    corrector_config: CorrectorConfig,
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
    runtime_binary_path: Path | None,
    runtime_host: str,
    verbose: bool,
) -> list[ConservativeCorrectionRecord]:
    if corrector_config.local_model_path is None or corrector_config.local_mmproj_path is None:
        raise RuntimeError("qwen-local correction requires explicit corrector model and mmproj paths")

    resolved_runtime = _resolve_qwen_runtime_overrides(corrector_config.local_runtime_overrides)
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
        profile=resolved_runtime.profile,
        binary_path=runtime_binary_path,
        host=runtime_host,
        port=resolved_runtime.port,
        threads=resolved_runtime.threads,
        threads_batch=resolved_runtime.threads_batch,
        ctx_size=resolved_runtime.ctx_size,
        n_predict=resolved_runtime.n_predict,
        reasoning=resolved_runtime.reasoning,
        gpu_layers=resolved_runtime.gpu_layers,
        no_mmproj_offload=resolved_runtime.no_mmproj_offload,
        startup_timeout_sec=resolved_runtime.startup_timeout_sec,
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
    key_to_unique_index: dict[Any, int] = {}
    unique_records: list[HybridDetectorRecord] = []
    record_to_unique_index: list[int] = []
    unique_group_sizes: list[int] = []

    for detector_record in detector_records:
        prepared = prepared_by_index[detector_record.index]
        key = (_prepared_identity_key(prepared), detector_record.shape)
        unique_index = key_to_unique_index.get(key)
        if unique_index is None:
            unique_index = len(unique_records)
            key_to_unique_index[key] = unique_index
            unique_records.append(detector_record)
            unique_group_sizes.append(0)
        unique_group_sizes[unique_index] += 1
        record_to_unique_index.append(unique_index)

    prompt_styles_by_unique_index = [
        corrector_prompt_for_shape(corrector_config, detector_record.shape)[1]
        for detector_record in unique_records
    ]
    unique_responses = _collect_gemini_correction_responses(
        unique_records=unique_records,
        prepared_by_index=prepared_by_index,
        corrector_config=corrector_config,
        verbose=verbose,
    )
    fallback_count = sum(1 for outcome in unique_responses if outcome.status == "fallback_baseline")
    if fallback_count:
        logger.warning(
            "Gemini corrector degraded to baseline for %d row(s) after request failures",
            fallback_count,
        )
    for detector_record, unique_index in zip(detector_records, record_to_unique_index, strict=True):
        outcome = unique_responses[unique_index]
        if outcome.status == "fallback_baseline":
            records.append(
                _build_failed_correction_record(
                    detector_record=detector_record,
                    corrector_name=corrector_name,
                    corrector_prompt_style=prompt_styles_by_unique_index[unique_index],
                    error_message=outcome.error_message,
                )
            )
            continue
        records.append(
            _build_correction_record(
                detector_record=detector_record,
                corrector_name=corrector_name,
                corrector_prompt_style=outcome.prompt_style,
                corrector_text=outcome.corrector_text,
                corrector_reasoning_content=outcome.reasoning_content,
            )
        )
    return records


def _collect_gemini_correction_responses(
    *,
    unique_records: list[HybridDetectorRecord],
    prepared_by_index: dict[int, _PreparedOCRInput],
    corrector_config: CorrectorConfig,
    verbose: bool,
) -> list[_GeminiCorrectionOutcome]:
    if not unique_records:
        return []

    worker_count = _gemini_parallel_worker_count(
        corrector_config=corrector_config,
        unique_record_count=len(unique_records),
    )
    if verbose:
        logger.info(
            "running Gemini corrector: unique_rows=%s workers=%s",
            len(unique_records),
            worker_count,
        )
    if worker_count < 2:
        return [
            _run_gemini_correction_task(
                detector_record=detector_record,
                prepared=prepared_by_index[detector_record.index],
                corrector_config=corrector_config,
                verbose=verbose,
                abort_event=None,
            )
            for detector_record in unique_records
        ]

    abort_event = threading.Event()
    unique_responses: list[_GeminiCorrectionOutcome | None] = [None] * len(unique_records)
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="istots-gemini") as executor:
        future_to_index = {
            executor.submit(
                _run_gemini_correction_task,
                detector_record=detector_record,
                prepared=prepared_by_index[detector_record.index],
                corrector_config=corrector_config,
                verbose=verbose,
                abort_event=abort_event,
            ): unique_index
            for unique_index, detector_record in enumerate(unique_records)
        }
        try:
            for future in as_completed(future_to_index):
                unique_index = future_to_index[future]
                unique_responses[unique_index] = future.result()
        except Exception:
            abort_event.set()
            for future in future_to_index:
                future.cancel()
            raise
    return [response for response in unique_responses if response is not None]


def _run_gemini_correction_task(
    *,
    detector_record: HybridDetectorRecord,
    prepared: _PreparedOCRInput,
    corrector_config: CorrectorConfig,
    verbose: bool,
    abort_event: threading.Event | None,
) -> _GeminiCorrectionOutcome:
    if verbose:
        logger.info(
            "running Gemini corrector: row=%s branch=%s shape=%s",
            detector_record.index,
            detector_record.detector_branch,
            detector_record.shape,
        )
    try:
        corrector_text, prompt_style, reasoning_content = request_gemini_correction(
            config=corrector_config,
            image=_prepared_image(prepared),
            shape=detector_record.shape,
            verbose=verbose,
            abort_event=abort_event,
        )
    except GeminiRequestFailedError as exc:
        return _GeminiCorrectionOutcome(
            corrector_text="",
            prompt_style=corrector_prompt_for_shape(corrector_config, detector_record.shape)[1],
            reasoning_content="",
            status="fallback_baseline",
            error_message=exc.reason,
        )
    return _GeminiCorrectionOutcome(
        corrector_text=corrector_text,
        prompt_style=prompt_style,
        reasoning_content=reasoning_content,
        status="applied",
    )


def _gemini_parallel_worker_count(
    *,
    corrector_config: CorrectorConfig,
    unique_record_count: int,
) -> int:
    max_workers = max(1, corrector_config.gemini_max_workers)
    if unique_record_count < max(1, corrector_config.gemini_parallel_min_rows):
        return 1
    return min(unique_record_count, max_workers)


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
        corrector_status="applied",
        corrector_error="",
    )


def _build_failed_correction_record(
    *,
    detector_record: HybridDetectorRecord,
    corrector_name: str,
    corrector_prompt_style: str,
    error_message: str,
) -> ConservativeCorrectionRecord:
    anchor_rows = build_focus_context(detector_record.baseline_text, detector_record.option_text)
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
        corrector_text="",
        conservative_merged_text=detector_record.baseline_text,
        applied_op_count=0,
        raw_changed=False,
        merged_changed=False,
        corrector_reasoning_content="",
        corrector_status="fallback_baseline",
        corrector_error=error_message,
    )


def _is_tall_subtitle_image(image: Image.Image) -> bool:
    width, height = image.size
    if width <= 0:
        return False
    return (float(height) / float(width)) >= TALL_SUBTITLE_RATIO_THRESHOLD


def _build_window_text_segment(prepared: _PreparedOCRInput, text: str) -> _WindowTextSegment | None:
    if not text:
        return None

    end = prepared.end
    if end <= prepared.start:
        end = prepared.start + timedelta(milliseconds=1)

    return _WindowTextSegment(
        start=prepared.start,
        end=end,
        text=text,
        window_id=prepared.window_id,
        left=prepared.left,
        top=prepared.top,
        right=prepared.right,
        bottom=prepared.bottom,
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
