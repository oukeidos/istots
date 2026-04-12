from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlamaOpenAISamplingRecipe:
    name: str
    temperature: float | None
    top_k: int | None
    top_p: float | None
    min_p: float | None
    presence_penalty: float | None
    frequency_penalty: float | None
    repeat_penalty: float | None


PADDLEOCR_LLAMA_OCR_EXPLICIT_DEFAULTS = LlamaOpenAISamplingRecipe(
    name="paddleocr_llama_ocr_explicit_defaults",
    temperature=0.0,
    top_k=40,
    top_p=0.95,
    min_p=0.05,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repeat_penalty=1.0,
)


def apply_openai_sampling_recipe(
    body: dict[str, Any],
    *,
    recipe: LlamaOpenAISamplingRecipe,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repeat_penalty: float | None = None,
    omit_temperature: bool = False,
) -> dict[str, float | int | None]:
    resolved = {
        "temperature": recipe.temperature if temperature is None else temperature,
        "top_k": recipe.top_k if top_k is None else top_k,
        "top_p": recipe.top_p if top_p is None else top_p,
        "min_p": recipe.min_p if min_p is None else min_p,
        "presence_penalty": (
            recipe.presence_penalty if presence_penalty is None else presence_penalty
        ),
        "frequency_penalty": (
            recipe.frequency_penalty if frequency_penalty is None else frequency_penalty
        ),
        "repeat_penalty": recipe.repeat_penalty if repeat_penalty is None else repeat_penalty,
    }
    if not omit_temperature and resolved["temperature"] is not None:
        body["temperature"] = float(resolved["temperature"])
    if resolved["top_k"] is not None:
        body["top_k"] = int(resolved["top_k"])
    if resolved["top_p"] is not None:
        body["top_p"] = float(resolved["top_p"])
    if resolved["min_p"] is not None:
        body["min_p"] = float(resolved["min_p"])
    if resolved["presence_penalty"] is not None:
        body["presence_penalty"] = float(resolved["presence_penalty"])
    if resolved["frequency_penalty"] is not None:
        body["frequency_penalty"] = float(resolved["frequency_penalty"])
    if resolved["repeat_penalty"] is not None:
        body["repeat_penalty"] = float(resolved["repeat_penalty"])
    return resolved
