from __future__ import annotations

from pathlib import Path

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
