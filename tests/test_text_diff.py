from __future__ import annotations

from istots.text_diff import GENERIC_TEXT_DIFF_PROFILE, assess_difference


def test_assess_difference_detects_exact_match() -> None:
    assessment = assess_difference("abc", "abc")
    assert assessment.label == "exact_match"
    assert assessment.meaningful is False


def test_assess_difference_detects_normalized_equivalence() -> None:
    assessment = assess_difference("A  B", "A B")
    assert assessment.label == "normalized_equivalent"
    assert assessment.meaningful is False


def test_assess_difference_detects_meaningful_difference() -> None:
    assessment = assess_difference("昴には", "昂には")
    assert assessment.label == "meaningful_difference"
    assert assessment.meaningful is True


def test_assess_difference_default_profile_keeps_japanese_orthographic_bucket() -> None:
    assessment = assess_difference("はっはっは", "はつはつは")
    assert assessment.label == "orthographic_equivalent"
    assert assessment.meaningful is False


def test_assess_difference_generic_profile_does_not_fold_japanese_ocr_drift() -> None:
    assessment = assess_difference("はっはっは", "はつはつは", profile=GENERIC_TEXT_DIFF_PROFILE)
    assert assessment.label == "meaningful_difference"
    assert assessment.meaningful is True
