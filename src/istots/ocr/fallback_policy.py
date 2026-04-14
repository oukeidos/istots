from __future__ import annotations

MAIN_OCR_BASELINE_PROMPT = "OCR:"
MAIN_OCR_LENGTH_FALLBACK_PROMPT = "Output only the exact text."
MAIN_OCR_LENGTH_FALLBACK_ROLES = frozenset({"ocr", "ocr-fast"})


def is_main_ocr_length_fallback_eligible(*, role: str, prompt_text: str) -> bool:
    return role in MAIN_OCR_LENGTH_FALLBACK_ROLES and prompt_text == MAIN_OCR_BASELINE_PROMPT
