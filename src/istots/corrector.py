from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import logging
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from PIL import Image

from istots.atomic_writer import atomic_write_json, atomic_write_jsonl
from istots.gemini_auth import resolve_gemini_api_key
from istots.ocr.types import Qwen35RuntimeOverrides
from istots.ocr.hf_backend import normalize_ocr_text

STRICT_OCR_V1_PROMPT = (
    "Transcribe only the visible subtitle text in the image. "
    "Output only the text. Preserve line breaks. Do not explain."
)
GENERAL_VERTICAL_HINT_V1_PROMPT = (
    STRICT_OCR_V1_PROMPT
    + "\n\n"
    + "Preserve the reading order shown in the image. "
    + "If the text is arranged vertically, follow the natural reading order of the vertical layout as shown. "
    + "Preserve line breaks only when the image clearly shows separate subtitle lines or separate speaker lines. "
    + "Do not introduce extra line breaks only because the text is vertical."
)
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_MAX_ATTEMPTS = 4
DEFAULT_GEMINI_RETRY_INITIAL_SLEEP_SEC = 1.0
DEFAULT_GEMINI_RETRY_MAX_SLEEP_SEC = 15.0
DEFAULT_GEMINI_REQUEST_TIMEOUT_SEC = 90.0
DEFAULT_GEMINI_MAX_WORKERS = 3
DEFAULT_GEMINI_PARALLEL_MIN_ROWS = 2
DEFAULT_GEMINI_RETRY_AFTER_CAP_SEC = 45.0
DEFAULT_GEMINI_CACHE_WAIT_POLL_SEC = 0.25
DEFAULT_GEMINI_CACHE_LEASE_STALE_SEC = 165.0
DEFAULT_RETRYABLE_GEMINI_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_GEMINI_CACHE_LEASE_OWNER_FILENAME = "owner.json"
LOCAL_QWEN_CTX_SIZE = 4096
LOCAL_QWEN_MAX_NEW_TOKENS = 128
logger = logging.getLogger(__name__)


class CorrectorMode(StrEnum):
    QWEN_LOCAL = "qwen-local"
    GEMINI = "gemini"


@dataclass(frozen=True)
class CorrectorConfig:
    mode: CorrectorMode
    output_path: Path | None = None
    local_model_path: Path | None = None
    local_mmproj_path: Path | None = None
    local_runtime_overrides: Qwen35RuntimeOverrides = field(default_factory=Qwen35RuntimeOverrides)
    api_key_env: str = "GEMINI_API_KEY"
    api_base: str = DEFAULT_GEMINI_API_BASE
    gemini_model: str = DEFAULT_GEMINI_MODEL
    thinking_level: str | None = "low"
    media_resolution: str | None = None
    temperature: float = 1.0
    cache_dir: Path | None = None
    max_attempts: int = DEFAULT_GEMINI_MAX_ATTEMPTS
    retry_initial_sleep: float = DEFAULT_GEMINI_RETRY_INITIAL_SLEEP_SEC
    retry_max_sleep: float = DEFAULT_GEMINI_RETRY_MAX_SLEEP_SEC
    request_timeout: float = DEFAULT_GEMINI_REQUEST_TIMEOUT_SEC
    gemini_retry_after_cap_sec: float = DEFAULT_GEMINI_RETRY_AFTER_CAP_SEC
    gemini_max_workers: int = DEFAULT_GEMINI_MAX_WORKERS
    gemini_parallel_min_rows: int = DEFAULT_GEMINI_PARALLEL_MIN_ROWS
    gemini_cache_wait_poll_sec: float = DEFAULT_GEMINI_CACHE_WAIT_POLL_SEC
    gemini_cache_lease_stale_sec: float = DEFAULT_GEMINI_CACHE_LEASE_STALE_SEC


@dataclass(frozen=True)
class ConservativeCorrectionRecord:
    index: int
    raw_index: int
    window_id: int
    start_ms: int
    end_ms: int
    detector_branch: str
    shape: str
    ratio: float
    option_role: str
    baseline_text: str
    option_text: str
    diff_label: str
    meaningful: bool
    char_error_rate: float
    anchor_count: int
    corrector_name: str
    corrector_prompt_style: str
    corrector_text: str
    conservative_merged_text: str
    applied_op_count: int
    raw_changed: bool
    merged_changed: bool
    corrector_reasoning_content: str = ""
    corrector_status: str = "applied"
    corrector_error: str = ""


class RetryableGeminiError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int | None,
        retry_after_sec: float | None,
        reason: str,
        response_body: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.retry_after_sec = retry_after_sec
        self.reason = reason
        self.response_body = response_body


class GeminiConfigurationError(RuntimeError):
    pass


class GeminiRequestFailedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _GeminiCacheLeasePaths:
    lease_dir: Path
    owner_path: Path


def write_correction_records(path: Path, records: list[ConservativeCorrectionRecord]) -> None:
    atomic_write_jsonl(path, (asdict(record) for record in records), ensure_ascii=False)


def corrector_prompt_for_shape(config: CorrectorConfig, shape: str) -> tuple[str, str]:
    if config.mode is CorrectorMode.GEMINI and shape == "tall":
        return GENERAL_VERTICAL_HINT_V1_PROMPT, "general_vertical_hint_v1"
    return STRICT_OCR_V1_PROMPT, "strict_ocr_v1"


def corrector_name_for_config(config: CorrectorConfig) -> str:
    if config.mode is CorrectorMode.QWEN_LOCAL:
        if config.local_model_path is None:
            raise RuntimeError("local corrector model path is required")
        return config.local_model_path.stem
    return config.gemini_model


def request_gemini_correction(
    *,
    config: CorrectorConfig,
    image: Image.Image,
    shape: str,
    verbose: bool = False,
    abort_event: threading.Event | None = None,
) -> tuple[str, str, str]:
    api_key, _ = resolve_gemini_api_key(config.api_key_env)
    if not api_key:
        raise GeminiConfigurationError(
            "missing Gemini API key. "
            f"Run `istots auth gemini set`, configure `istots auth gemini env-file set PATH`, "
            f"or export {config.api_key_env}."
        )

    prompt, prompt_style = corrector_prompt_for_shape(config, shape)
    payload = _request_gemini_one(
        api_key=api_key,
        api_base=config.api_base,
        model=config.gemini_model,
        prompt=prompt,
        image=image,
        thinking_level=config.thinking_level,
        media_resolution=config.media_resolution,
        temperature=config.temperature,
        cache_dir=config.cache_dir,
        max_attempts=config.max_attempts,
        retry_initial_sleep=config.retry_initial_sleep,
        retry_max_sleep=config.retry_max_sleep,
        request_timeout=config.request_timeout,
        retry_after_cap_sec=config.gemini_retry_after_cap_sec,
        cache_wait_poll_sec=config.gemini_cache_wait_poll_sec,
        cache_lease_stale_sec=config.gemini_cache_lease_stale_sec,
        verbose=verbose,
        abort_event=abort_event,
    )
    return normalize_ocr_text(payload["text"]), prompt_style, str(payload.get("reasoning_content", ""))


def _image_to_inline_data(image: Image.Image) -> tuple[dict[str, object], bytes]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return (
        {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(image_bytes).decode("ascii"),
            },
        },
        image_bytes,
    )


def _cache_request_key(
    *,
    model: str,
    prompt: str,
    thinking_level: str | None,
    media_resolution: str | None,
    temperature: float,
    image_bytes: bytes,
) -> tuple[str, dict[str, str | float | None]]:
    request_meta: dict[str, str | float | None] = {
        "model": model,
        "prompt": prompt,
        "thinking_level": thinking_level,
        "media_resolution": media_resolution,
        "temperature": temperature,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(request_meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, request_meta


def _load_cached_item(cache_dir: Path | None, cache_key: str) -> dict | None:
    if cache_dir is None:
        return None
    cache_path = cache_dir / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _save_cached_item(cache_dir: Path | None, cache_key: str, payload: dict) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    atomic_write_json(cache_path, payload, ensure_ascii=False, indent=2)


def _cache_lease_paths(cache_dir: Path | None, cache_key: str) -> _GeminiCacheLeasePaths | None:
    if cache_dir is None:
        return None
    lease_dir = cache_dir / f".{cache_key}.lease"
    return _GeminiCacheLeasePaths(
        lease_dir=lease_dir,
        owner_path=lease_dir / DEFAULT_GEMINI_CACHE_LEASE_OWNER_FILENAME,
    )


def _load_cache_lease_owner(paths: _GeminiCacheLeasePaths) -> dict[str, object] | None:
    if not paths.owner_path.exists():
        return None
    try:
        payload = json.loads(paths.owner_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _atomic_write_cache_lease_owner(
    paths: _GeminiCacheLeasePaths,
    *,
    cache_key: str,
    instance_id: str,
    created_at: float,
) -> None:
    now = time.time()
    atomic_write_json(
        paths.owner_path,
        {
            "pid": os.getpid(),
            "cache_key": cache_key,
            "instance_id": instance_id,
            "created_at": created_at,
            "updated_at": now,
        },
        ensure_ascii=False,
        indent=2,
    )


def _touch_cache_lease_owner(
    paths: _GeminiCacheLeasePaths | None,
    *,
    cache_key: str,
    instance_id: str | None,
    created_at: float | None,
) -> None:
    if paths is None or instance_id is None or created_at is None:
        return
    _atomic_write_cache_lease_owner(
        paths,
        cache_key=cache_key,
        instance_id=instance_id,
        created_at=created_at,
    )


def _remove_cache_lease_dir(paths: _GeminiCacheLeasePaths) -> None:
    try:
        paths.owner_path.unlink()
    except FileNotFoundError:
        pass
    try:
        paths.lease_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        for child in paths.lease_dir.iterdir():
            if child.is_dir():
                continue
            try:
                child.unlink()
            except OSError:
                return
        try:
            paths.lease_dir.rmdir()
        except OSError:
            return


def _release_cache_lease(
    paths: _GeminiCacheLeasePaths | None,
    *,
    instance_id: str | None,
) -> None:
    if paths is None or instance_id is None:
        return
    owner = _load_cache_lease_owner(paths)
    if owner is not None and str(owner.get("instance_id")) != instance_id:
        return
    _remove_cache_lease_dir(paths)


def _cache_lease_is_stale(
    owner: dict[str, object] | None,
    *,
    stale_after_sec: float,
) -> bool:
    if owner is None:
        return True
    updated_at = owner.get("updated_at", owner.get("created_at"))
    try:
        updated_at_float = float(updated_at)
    except (TypeError, ValueError):
        return True
    return (time.time() - updated_at_float) > max(0.0, stale_after_sec)


def _try_acquire_cache_lease(
    paths: _GeminiCacheLeasePaths | None,
    *,
    cache_key: str,
) -> tuple[str | None, float | None]:
    if paths is None:
        return None, None
    try:
        paths.lease_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None, None
    instance_id = uuid.uuid4().hex
    created_at = time.time()
    try:
        _atomic_write_cache_lease_owner(
            paths,
            cache_key=cache_key,
            instance_id=instance_id,
            created_at=created_at,
        )
    except Exception:
        _remove_cache_lease_dir(paths)
        raise
    return instance_id, created_at


def _abort_requested(abort_event: threading.Event | None) -> bool:
    return abort_event.is_set() if abort_event is not None else False


def _extract_text_and_thoughts(response_body: dict) -> tuple[str, str]:
    text_parts: list[str] = []
    thought_parts: list[str] = []
    for candidate in response_body.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if not text:
                continue
            if part.get("thought"):
                thought_parts.append(text)
            else:
                text_parts.append(text)
    return "".join(text_parts).strip(), "".join(thought_parts).strip()


def _parse_retry_after(headers: object) -> float | None:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
    except Exception:
        value = None
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = dt.datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
        now = dt.datetime.utcnow()
        return max(0.0, (parsed - now).total_seconds())
    except ValueError:
        return None


def _parse_retry_delay_from_error_body(body_text: str | None) -> float | None:
    if not body_text:
        return None
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError:
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    details = error.get("details")
    if not isinstance(details, list):
        return None
    for item in details:
        if not isinstance(item, dict):
            continue
        retry_delay = item.get("retryDelay")
        if retry_delay is None:
            continue
        value = str(retry_delay).strip()
        if value.endswith("s"):
            value = value[:-1]
        try:
            return max(0.0, float(value))
        except ValueError:
            continue
    return None


def _request_gemini_one_once(
    *,
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    image: Image.Image,
    thinking_level: str | None,
    media_resolution: str | None,
    temperature: float,
    request_timeout: float,
    retry_after_cap_sec: float,
) -> tuple[dict, float]:
    image_part, image_bytes = _image_to_inline_data(image)
    cache_key, request_meta = _cache_request_key(
        model=model,
        prompt=prompt,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        temperature=temperature,
        image_bytes=image_bytes,
    )
    if media_resolution is not None:
        image_part["media_resolution"] = {"level": media_resolution}
    body: dict = {
        "contents": [
            {
                "parts": [
                    image_part,
                    {"text": prompt},
                ]
            }
        ]
    }
    generation_config: dict[str, object] = {"temperature": temperature}
    if thinking_level is not None:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    body["generationConfig"] = generation_config
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base}/{model}:generateContent",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        if exc.code in DEFAULT_RETRYABLE_GEMINI_STATUS_CODES:
            retry_after = _parse_retry_after(exc.headers)
            body_retry = _parse_retry_delay_from_error_body(body_text)
            retry_after = max(
                retry_after if retry_after is not None else 0.0,
                body_retry if body_retry is not None else 0.0,
            )
            if retry_after > 0:
                retry_after = min(retry_after, max(0.0, retry_after_cap_sec))
            raise RetryableGeminiError(
                status_code=exc.code,
                retry_after_sec=retry_after if retry_after > 0 else None,
                reason=f"http_{exc.code}",
                response_body=body_text,
            ) from exc
        raise
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise RetryableGeminiError(
            status_code=None,
            retry_after_sec=None,
            reason=type(exc).__name__,
            response_body=None,
        ) from exc

    elapsed = time.perf_counter() - started
    text, thoughts = _extract_text_and_thoughts(response_body)
    return (
        {
            "elapsed_sec": elapsed,
            "text": text,
            "reasoning_content": thoughts,
            "response": response_body,
            "cache_hit": False,
            "cache_key": cache_key,
            "request_meta": request_meta,
        },
        elapsed,
    )


def _request_gemini_one(
    *,
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    image: Image.Image,
    thinking_level: str | None,
    media_resolution: str | None,
    temperature: float,
    cache_dir: Path | None,
    max_attempts: int,
    retry_initial_sleep: float,
    retry_max_sleep: float,
    request_timeout: float,
    retry_after_cap_sec: float,
    cache_wait_poll_sec: float,
    cache_lease_stale_sec: float,
    verbose: bool,
    abort_event: threading.Event | None,
) -> dict:
    image_part, image_bytes = _image_to_inline_data(image)
    del image_part
    cache_key, request_meta = _cache_request_key(
        model=model,
        prompt=prompt,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        temperature=temperature,
        image_bytes=image_bytes,
    )
    cached = _load_cached_item(cache_dir, cache_key)
    if cached is not None:
        return {
            "elapsed_sec": float(cached["elapsed_sec"]),
            "text": str(cached["text"]),
            "reasoning_content": str(cached.get("reasoning_content", "")),
            "response": cached["response"],
            "cache_hit": True,
            "cache_key": cache_key,
            "request_meta": request_meta,
        }

    lease_paths = _cache_lease_paths(cache_dir, cache_key)
    lease_instance_id: str | None = None
    lease_created_at: float | None = None
    while cache_dir is not None and lease_instance_id is None:
        if _abort_requested(abort_event):
            raise RuntimeError("Gemini correction aborted")
        cached = _load_cached_item(cache_dir, cache_key)
        if cached is not None:
            return {
                "elapsed_sec": float(cached["elapsed_sec"]),
                "text": str(cached["text"]),
                "reasoning_content": str(cached.get("reasoning_content", "")),
                "response": cached["response"],
                "cache_hit": True,
                "cache_key": cache_key,
                "request_meta": request_meta,
            }
        owner = _load_cache_lease_owner(lease_paths) if lease_paths is not None else None
        lease_needs_reclaim = lease_paths is not None and lease_paths.lease_dir.exists() and (
            owner is None
            or _cache_lease_is_stale(
                owner,
                stale_after_sec=cache_lease_stale_sec,
            )
        )
        if lease_needs_reclaim:
            if verbose:
                logger.info(
                    "reclaiming stale Gemini cache lease: cache_key=%s owner_pid=%s",
                    cache_key,
                    owner.get("pid") if owner is not None else None,
                )
            _remove_cache_lease_dir(lease_paths)
            owner = None
        lease_instance_id, lease_created_at = _try_acquire_cache_lease(
            lease_paths,
            cache_key=cache_key,
        )
        if lease_instance_id is not None:
            break
        if verbose:
            logger.info("waiting for active Gemini cache lease: cache_key=%s", cache_key)
        time.sleep(max(0.0, cache_wait_poll_sec))

    sleep_sec = retry_initial_sleep
    last_error: RetryableGeminiError | None = None
    try:
        if cache_dir is not None:
            cached = _load_cached_item(cache_dir, cache_key)
            if cached is not None:
                return {
                    "elapsed_sec": float(cached["elapsed_sec"]),
                    "text": str(cached["text"]),
                    "reasoning_content": str(cached.get("reasoning_content", "")),
                    "response": cached["response"],
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "request_meta": request_meta,
                }
        for attempt in range(1, max(1, max_attempts) + 1):
            if _abort_requested(abort_event):
                raise RuntimeError("Gemini correction aborted")
            try:
                _touch_cache_lease_owner(
                    lease_paths,
                    cache_key=cache_key,
                    instance_id=lease_instance_id,
                    created_at=lease_created_at,
                )
                payload, _ = _request_gemini_one_once(
                    api_key=api_key,
                    api_base=api_base,
                    model=model,
                    prompt=prompt,
                    image=image,
                    thinking_level=thinking_level,
                    media_resolution=media_resolution,
                    temperature=temperature,
                    request_timeout=request_timeout,
                    retry_after_cap_sec=retry_after_cap_sec,
                )
                _save_cached_item(cache_dir, cache_key, payload)
                return payload
            except RetryableGeminiError as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                if exc.retry_after_sec is not None:
                    delay_sec = min(retry_max_sleep, max(0.0, exc.retry_after_sec))
                else:
                    delay_cap = min(retry_max_sleep, max(0.0, sleep_sec))
                    delay_sec = random.uniform(0.0, delay_cap)
                if verbose:
                    logger.info(
                        "retrying Gemini request: cache_key=%s attempt=%s/%s reason=%s delay=%.3fs",
                        cache_key,
                        attempt + 1,
                        max(1, max_attempts),
                        exc.reason,
                        delay_sec,
                    )
                _touch_cache_lease_owner(
                    lease_paths,
                    cache_key=cache_key,
                    instance_id=lease_instance_id,
                    created_at=lease_created_at,
                )
                time.sleep(delay_sec)
                sleep_sec = min(retry_max_sleep, max(sleep_sec * 2.0, retry_initial_sleep))
    finally:
        _release_cache_lease(
            lease_paths,
            instance_id=lease_instance_id,
        )

    if last_error is None:
        raise RuntimeError("Gemini request failed without an explicit error")
    raise GeminiRequestFailedError(last_error.reason) from last_error
