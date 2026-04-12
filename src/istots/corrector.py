from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image

from istots.gemini_auth import resolve_gemini_api_key
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
LOCAL_QWEN_CTX_SIZE = 4096
LOCAL_QWEN_MAX_NEW_TOKENS = 128


class CorrectorMode(StrEnum):
    QWEN_LOCAL = "qwen-local"
    GEMINI = "gemini"


@dataclass(frozen=True)
class CorrectorConfig:
    mode: CorrectorMode
    output_path: Path | None = None
    local_model_path: Path | None = None
    local_mmproj_path: Path | None = None
    local_no_mmproj_offload: bool = False
    port: int | None = None
    startup_timeout_sec: float = 120.0
    api_key_env: str = "GEMINI_API_KEY"
    api_base: str = DEFAULT_GEMINI_API_BASE
    gemini_model: str = DEFAULT_GEMINI_MODEL
    thinking_level: str | None = "low"
    media_resolution: str | None = None
    temperature: float = 1.0
    cache_dir: Path | None = None
    max_attempts: int = 6
    retry_initial_sleep: float = 2.0
    retry_max_sleep: float = 30.0
    request_timeout: float = 180.0


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


def write_correction_records(path: Path, records: list[ConservativeCorrectionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


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
) -> tuple[str, str, str]:
    api_key, _ = resolve_gemini_api_key(config.api_key_env)
    if not api_key:
        raise RuntimeError(
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
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cached_item(cache_dir: Path | None, cache_key: str, payload: dict) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        if exc.code in {429, 500, 503, 504}:
            retry_after = _parse_retry_after(exc.headers)
            body_retry = _parse_retry_delay_from_error_body(body_text)
            retry_after = max(
                retry_after if retry_after is not None else 0.0,
                body_retry if body_retry is not None else 0.0,
            )
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

    sleep_sec = retry_initial_sleep
    last_error: RetryableGeminiError | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
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
            )
            _save_cached_item(cache_dir, cache_key, payload)
            return payload
        except RetryableGeminiError as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            retry_after = exc.retry_after_sec if exc.retry_after_sec is not None else sleep_sec
            jittered = retry_after * random.uniform(0.9, 1.1)
            time.sleep(min(retry_max_sleep, max(0.0, jittered)))
            sleep_sec = min(retry_max_sleep, max(sleep_sec * 2.0, retry_initial_sleep))

    if last_error is None:
        raise RuntimeError("Gemini request failed without an explicit error")
    raise RuntimeError(f"Gemini correction failed after retries: {last_error.reason}") from last_error
