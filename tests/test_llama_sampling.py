from __future__ import annotations

from istots.llama_sampling import (
    PADDLEOCR_LLAMA_OCR_EXPLICIT_DEFAULTS,
    apply_openai_sampling_recipe,
)


def test_apply_openai_sampling_recipe_applies_explicit_defaults() -> None:
    body: dict[str, object] = {}
    resolved = apply_openai_sampling_recipe(
        body,
        recipe=PADDLEOCR_LLAMA_OCR_EXPLICIT_DEFAULTS,
    )

    assert resolved["temperature"] == 0.0
    assert body == {
        "temperature": 0.0,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.0,
    }
