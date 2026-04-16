from __future__ import annotations

import io
import json
from pathlib import Path
import urllib.error

import pytest
from PIL import Image

from istots import corrector
from istots.corrector import CorrectorConfig, CorrectorMode, corrector_prompt_for_shape


def test_corrector_prompt_for_shape_uses_vertical_hint_for_gemini_tall_rows() -> None:
    prompt, style = corrector_prompt_for_shape(
        CorrectorConfig(mode=CorrectorMode.GEMINI),
        "tall",
    )

    assert style == "general_vertical_hint_v1"
    assert "If the text is arranged vertically" in prompt


def test_corrector_prompt_for_shape_keeps_strict_prompt_for_local_qwen() -> None:
    prompt, style = corrector_prompt_for_shape(
        CorrectorConfig(
            mode=CorrectorMode.QWEN_LOCAL,
            local_model_path=Path("/tmp/qwen.gguf"),
            local_mmproj_path=Path("/tmp/qwen-mmproj.gguf"),
        ),
        "tall",
    )

    assert style == "strict_ocr_v1"
    assert "If the text is arranged vertically" not in prompt


def test_image_to_inline_data_wraps_png_payload() -> None:
    payload, image_bytes = corrector._image_to_inline_data(Image.new("RGB", (2, 3), "white"))  # noqa: SLF001

    assert image_bytes.startswith(b"\x89PNG")
    assert payload["inline_data"]["mime_type"] == "image/png"
    assert isinstance(payload["inline_data"]["data"], str)


def test_request_gemini_one_once_sends_inline_image_part(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "ok",
                                    }
                                ]
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(corrector.urllib.request, "urlopen", fake_urlopen)

    payload, _ = corrector._request_gemini_one_once(  # noqa: SLF001
        api_key="test-key",
        api_base="https://example.test/models",
        model="gemini-test",
        prompt="prompt-text",
        image=Image.new("RGB", (2, 3), "white"),
        thinking_level="low",
        media_resolution=None,
        temperature=1.0,
        request_timeout=30.0,
        retry_after_cap_sec=12.0,
    )

    body = captured["body"]
    assert captured["url"] == "https://example.test/models/gemini-test:generateContent"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["X-goog-api-key"] == "test-key"
    assert body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/png"
    assert body["contents"][0]["parts"][1] == {"text": "prompt-text"}
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert payload["text"] == "ok"


def test_request_gemini_one_once_caps_retry_after_hint(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "unavailable",
            {"Retry-After": "999"},
            io.BytesIO(b'{"error":{"details":[{"retryDelay":"1000s"}]}}'),
        )

    monkeypatch.setattr(corrector.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(corrector.RetryableGeminiError) as excinfo:
        corrector._request_gemini_one_once(  # noqa: SLF001
            api_key="test-key",
            api_base="https://example.test/models",
            model="gemini-test",
            prompt="prompt-text",
            image=Image.new("RGB", (2, 3), "white"),
            thinking_level="low",
            media_resolution=None,
            temperature=1.0,
            request_timeout=30.0,
            retry_after_cap_sec=12.0,
        )

    assert excinfo.value.retry_after_sec == 12.0


def test_request_gemini_one_retries_retryable_errors(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def fake_request_once(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise corrector.RetryableGeminiError(
                status_code=503,
                retry_after_sec=None,
                reason="http_503",
            )
        return (
            {
                "elapsed_sec": 0.1,
                "text": "ok",
                "reasoning_content": "",
                "response": {"candidates": []},
                "cache_hit": False,
                "cache_key": "ignored",
                "request_meta": {},
            },
            0.1,
        )

    monkeypatch.setattr(corrector, "_request_gemini_one_once", fake_request_once)  # noqa: SLF001
    monkeypatch.setattr(corrector.random, "uniform", lambda low, high: 1.5)
    monkeypatch.setattr(corrector.time, "sleep", sleep_calls.append)

    payload = corrector._request_gemini_one(  # noqa: SLF001
        api_key="test-key",
        api_base="https://example.test/models",
        model="gemini-test",
        prompt="prompt-text",
        image=Image.new("RGB", (2, 3), "white"),
        thinking_level="low",
        media_resolution=None,
        temperature=1.0,
        cache_dir=tmp_path / "cache",
        max_attempts=2,
        retry_initial_sleep=2.0,
        retry_max_sleep=30.0,
        request_timeout=30.0,
        retry_after_cap_sec=12.0,
        cache_wait_poll_sec=0.01,
        cache_lease_stale_sec=60.0,
        verbose=False,
        abort_event=None,
    )

    assert attempts["count"] == 2
    assert sleep_calls == [1.5]
    assert payload["text"] == "ok"


def test_request_gemini_one_waits_for_existing_cache_lease(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    image = Image.new("RGB", (2, 3), "white")
    image_bytes = corrector._image_to_inline_data(image)[1]  # noqa: SLF001
    cache_key, _ = corrector._cache_request_key(  # noqa: SLF001
        model="gemini-test",
        prompt="prompt-text",
        thinking_level="low",
        media_resolution=None,
        temperature=1.0,
        image_bytes=image_bytes,
    )
    lease_paths = corrector._cache_lease_paths(cache_dir, cache_key)  # noqa: SLF001
    assert lease_paths is not None
    lease_paths.lease_dir.mkdir()
    corrector.atomic_write_json(
        lease_paths.owner_path,
        {
            "pid": 99999,
            "cache_key": cache_key,
            "instance_id": "foreign",
            "created_at": corrector.time.time(),
            "updated_at": corrector.time.time(),
        },
        ensure_ascii=False,
        indent=2,
    )

    def fake_sleep(delay: float) -> None:
        corrector._save_cached_item(  # noqa: SLF001
            cache_dir,
            cache_key,
            {
                "elapsed_sec": 0.2,
                "text": "cached-result",
                "reasoning_content": "",
                "response": {"candidates": []},
                "cache_hit": False,
                "cache_key": cache_key,
                "request_meta": {},
            },
        )

    monkeypatch.setattr(corrector.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        corrector,
        "_request_gemini_one_once",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("live request should not run")),
    )

    payload = corrector._request_gemini_one(  # noqa: SLF001
        api_key="test-key",
        api_base="https://example.test/models",
        model="gemini-test",
        prompt="prompt-text",
        image=image,
        thinking_level="low",
        media_resolution=None,
        temperature=1.0,
        cache_dir=cache_dir,
        max_attempts=2,
        retry_initial_sleep=2.0,
        retry_max_sleep=30.0,
        request_timeout=30.0,
        retry_after_cap_sec=12.0,
        cache_wait_poll_sec=0.01,
        cache_lease_stale_sec=60.0,
        verbose=False,
        abort_event=None,
    )

    assert payload["cache_hit"] is True
    assert payload["text"] == "cached-result"


def test_request_gemini_one_reclaims_stale_cache_lease(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    image = Image.new("RGB", (2, 3), "white")
    image_bytes = corrector._image_to_inline_data(image)[1]  # noqa: SLF001
    cache_key, _ = corrector._cache_request_key(  # noqa: SLF001
        model="gemini-test",
        prompt="prompt-text",
        thinking_level="low",
        media_resolution=None,
        temperature=1.0,
        image_bytes=image_bytes,
    )
    lease_paths = corrector._cache_lease_paths(cache_dir, cache_key)  # noqa: SLF001
    assert lease_paths is not None
    lease_paths.lease_dir.mkdir()
    corrector.atomic_write_json(
        lease_paths.owner_path,
        {
            "pid": 99999,
            "cache_key": cache_key,
            "instance_id": "stale-owner",
            "created_at": corrector.time.time() - 1000.0,
            "updated_at": corrector.time.time() - 1000.0,
        },
        ensure_ascii=False,
        indent=2,
    )

    monkeypatch.setattr(
        corrector,
        "_request_gemini_one_once",
        lambda **kwargs: (
            {
                "elapsed_sec": 0.2,
                "text": "live-result",
                "reasoning_content": "",
                "response": {"candidates": []},
                "cache_hit": False,
                "cache_key": cache_key,
                "request_meta": {},
            },
            0.2,
        ),
    )

    payload = corrector._request_gemini_one(  # noqa: SLF001
        api_key="test-key",
        api_base="https://example.test/models",
        model="gemini-test",
        prompt="prompt-text",
        image=image,
        thinking_level="low",
        media_resolution=None,
        temperature=1.0,
        cache_dir=cache_dir,
        max_attempts=2,
        retry_initial_sleep=2.0,
        retry_max_sleep=30.0,
        request_timeout=30.0,
        retry_after_cap_sec=12.0,
        cache_wait_poll_sec=0.01,
        cache_lease_stale_sec=1.0,
        verbose=False,
        abort_event=None,
    )

    assert payload["text"] == "live-result"
    assert not lease_paths.lease_dir.exists()
