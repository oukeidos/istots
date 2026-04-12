from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_DASH_EQUIVALENTS = {
    ord("‐"): "-",
    ord("‑"): "-",
    ord("‒"): "-",
    ord("–"): "-",
    ord("—"): "-",
    ord("―"): "-",
    ord("ー"): "-",
    ord("ｰ"): "-",
    ord("─"): "-",
    ord("━"): "-",
    ord("〜"): "~",
    ord("～"): "~",
}
_WHITESPACE_RE = re.compile(r"\s+")
_KANA_RE = re.compile(r"[ぁ-ゖァ-ヶ]")
_GEMINATE_FOLLOW_RE = re.compile(r"[かきくけこさしすせそたちつてとぱぴぷぺぽカキクケコサシスセソタチツテトパピプペポ]")
_SMALL_KANA_EQUIVALENTS = str.maketrans(
    {
        "ぁ": "あ",
        "ぃ": "い",
        "ぅ": "う",
        "ぇ": "え",
        "ぉ": "お",
        "っ": "つ",
        "ゃ": "や",
        "ゅ": "ゆ",
        "ょ": "よ",
        "ゎ": "わ",
        "ゕ": "か",
        "ゖ": "け",
        "ァ": "ア",
        "ィ": "イ",
        "ゥ": "ウ",
        "ェ": "エ",
        "ォ": "オ",
        "ッ": "ツ",
        "ャ": "ヤ",
        "ュ": "ユ",
        "ョ": "ヨ",
        "ヮ": "ワ",
        "ヵ": "カ",
        "ヶ": "ケ",
    }
)


@dataclass(frozen=True)
class DiffAssessment:
    label: str
    meaningful: bool
    raw_equal: bool
    canonical_equal: bool
    punctuation_equal: bool
    orthographic_equal: bool
    char_error_rate: float
    edit_distance: int
    reference_canonical: str
    candidate_canonical: str
    note: str


def canonicalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.translate(_DASH_EQUIVALENTS)
    lines = [" ".join(part for part in line.split()) for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def equivalence_text(text: str) -> str:
    normalized = canonicalize_text(text)
    normalized = _WHITESPACE_RE.sub("", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized


def punctuation_fold_text(text: str) -> str:
    folded: list[str] = []
    for char in equivalence_text(text):
        category = unicodedata.category(char)
        if category and category[0] in {"P", "S", "Z"}:
            continue
        folded.append(char)
    return "".join(folded)


def _looks_like_trailing_long_vowel_confusion(text: str, index: int) -> bool:
    if text[index] != "一" or index <= 0:
        return False
    previous = text[index - 1]
    if not _KANA_RE.fullmatch(previous):
        return False
    next_index = index + 1
    if next_index >= len(text):
        return True
    next_char = text[next_index]
    if next_char == "\n":
        return True
    category = unicodedata.category(next_char)
    return bool(category and category[0] in {"P", "S", "Z"})


def orthographic_fold_text(text: str) -> str:
    normalized = equivalence_text(text)
    chars: list[str] = []
    for index, char in enumerate(normalized):
        if char == "一" and _looks_like_trailing_long_vowel_confusion(normalized, index):
            chars.append("-")
            continue
        chars.append(char)
    normalized = "".join(chars).translate(_SMALL_KANA_EQUIVALENTS)
    normalized = re.sub(r"つ(?=" + _GEMINATE_FOLLOW_RE.pattern + r")", "", normalized)
    normalized = re.sub(r"ツ(?=[カキクケコサシスセソタチツテトパピプペポ])", "", normalized)
    return normalized


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    current = [0] * (len(right) + 1)
    for left_index, left_char in enumerate(left, start=1):
        current[0] = left_index
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (0 if left_char == right_char else 1)
            current[right_index] = min(insertion, deletion, substitution)
        previous, current = current, previous
    return previous[-1]


def assess_difference(reference: str, candidate: str) -> DiffAssessment:
    reference = reference or ""
    candidate = candidate or ""
    raw_equal = reference == candidate
    canonical_reference = equivalence_text(reference)
    canonical_candidate = equivalence_text(candidate)
    canonical_equal = canonical_reference == canonical_candidate
    punctuation_reference = punctuation_fold_text(reference)
    punctuation_candidate = punctuation_fold_text(candidate)
    punctuation_equal = punctuation_reference == punctuation_candidate
    orthographic_reference = orthographic_fold_text(reference)
    orthographic_candidate = orthographic_fold_text(candidate)
    orthographic_equal = orthographic_reference == orthographic_candidate
    edit_distance = levenshtein_distance(canonical_reference, canonical_candidate)
    denom = max(len(canonical_reference), 1)
    char_error_rate = edit_distance / denom

    if raw_equal:
        return DiffAssessment(
            label="exact_match",
            meaningful=False,
            raw_equal=True,
            canonical_equal=True,
            punctuation_equal=True,
            orthographic_equal=True,
            char_error_rate=0.0,
            edit_distance=0,
            reference_canonical=canonical_reference,
            candidate_canonical=canonical_candidate,
            note="raw text is identical",
        )
    if canonical_equal:
        return DiffAssessment(
            label="normalized_equivalent",
            meaningful=False,
            raw_equal=False,
            canonical_equal=True,
            punctuation_equal=True,
            orthographic_equal=True,
            char_error_rate=0.0,
            edit_distance=0,
            reference_canonical=canonical_reference,
            candidate_canonical=canonical_candidate,
            note="difference is limited to width, whitespace, or dash normalization",
        )
    if punctuation_equal:
        return DiffAssessment(
            label="punctuation_equivalent",
            meaningful=False,
            raw_equal=False,
            canonical_equal=False,
            punctuation_equal=True,
            orthographic_equal=True,
            char_error_rate=char_error_rate,
            edit_distance=edit_distance,
            reference_canonical=canonical_reference,
            candidate_canonical=canonical_candidate,
            note="difference disappears after removing punctuation and spacing",
        )
    if orthographic_equal:
        return DiffAssessment(
            label="orthographic_equivalent",
            meaningful=False,
            raw_equal=False,
            canonical_equal=False,
            punctuation_equal=False,
            orthographic_equal=True,
            char_error_rate=char_error_rate,
            edit_distance=edit_distance,
            reference_canonical=canonical_reference,
            candidate_canonical=canonical_candidate,
            note="difference is limited to Japanese small-kana, sokuon, or trailing long-vowel OCR drift",
        )
    return DiffAssessment(
        label="meaningful_difference",
        meaningful=True,
        raw_equal=False,
        canonical_equal=False,
        punctuation_equal=False,
        orthographic_equal=False,
        char_error_rate=char_error_rate,
        edit_distance=edit_distance,
        reference_canonical=canonical_reference,
        candidate_canonical=canonical_candidate,
        note="character-level content differs after normalization",
    )
