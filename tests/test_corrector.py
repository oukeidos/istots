from __future__ import annotations

import json
from pathlib import Path

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
    )

    body = captured["body"]
    assert captured["url"] == "https://example.test/models/gemini-test:generateContent"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["X-goog-api-key"] == "test-key"
    assert body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/png"
    assert body["contents"][0]["parts"][1] == {"text": "prompt-text"}
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert payload["text"] == "ok"
