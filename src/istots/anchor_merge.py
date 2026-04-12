from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AnchorRow:
    index: int
    tag: str
    baseline_span: tuple[int, int]
    option_span: tuple[int, int]
    baseline_frag: str
    option_frag: str


@dataclass(frozen=True)
class AppliedMergeOp:
    index: int
    tag: str
    baseline_span: tuple[int, int]
    candidate_span: tuple[int, int]
    baseline_frag: str
    candidate_frag: str


@dataclass(frozen=True)
class AnchorMergeResult:
    merged_text: str
    diff_ranges: tuple[tuple[int, int], ...]
    diff_points: tuple[int, ...]
    applied_ops: tuple[AppliedMergeOp, ...]


def iter_diff_ops(left: str, right: str) -> Iterable[tuple[str, int, int, int, int, str, str]]:
    matcher = difflib.SequenceMatcher(a=left, b=right)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        yield tag, i1, i2, j1, j2, left[i1:i2], right[j1:j2]


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((s, e) for s, e in ranges if s < e):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def overlaps_ambiguity(
    start: int,
    end: int,
    diff_ranges: list[tuple[int, int]],
    diff_points: set[int],
) -> bool:
    if start == end:
        if start in diff_points:
            return True
        return any(range_start <= start < range_end for range_start, range_end in diff_ranges)
    if any(max(start, range_start) < min(end, range_end) for range_start, range_end in diff_ranges):
        return True
    return any(start <= point <= end for point in diff_points)


def build_focus_context(baseline: str, option: str) -> tuple[AnchorRow, ...]:
    rows: list[AnchorRow] = []
    for index, (tag, i1, i2, j1, j2, baseline_frag, option_frag) in enumerate(
        iter_diff_ops(baseline, option),
        start=1,
    ):
        rows.append(
            AnchorRow(
                index=index,
                tag=tag,
                baseline_span=(i1, i2),
                option_span=(j1, j2),
                baseline_frag=baseline_frag,
                option_frag=option_frag,
            )
        )
    return tuple(rows)


def build_union_anchor_mask(anchor_rows: Iterable[AnchorRow]) -> tuple[list[tuple[int, int]], set[int]]:
    ranges: list[tuple[int, int]] = []
    points: set[int] = set()
    for row in anchor_rows:
        start, end = row.baseline_span
        if start < end:
            ranges.append((start, end))
        else:
            points.add(start)
    return merge_ranges(ranges), points


def apply_union_anchor_merge(
    baseline_text: str,
    candidate_text: str,
    anchor_rows: Iterable[AnchorRow],
) -> AnchorMergeResult:
    diff_ranges, diff_points = build_union_anchor_mask(anchor_rows)
    replacements: list[tuple[int, int, str]] = []
    applied_ops: list[AppliedMergeOp] = []
    for op_index, (tag, i1, i2, j1, j2, baseline_frag, candidate_frag) in enumerate(
        iter_diff_ops(baseline_text, candidate_text),
        start=1,
    ):
        if not overlaps_ambiguity(i1, i2, diff_ranges, diff_points):
            continue
        replacements.append((i1, i2, candidate_frag))
        applied_ops.append(
            AppliedMergeOp(
                index=op_index,
                tag=tag,
                baseline_span=(i1, i2),
                candidate_span=(j1, j2),
                baseline_frag=baseline_frag,
                candidate_frag=candidate_frag,
            )
        )

    merged = baseline_text
    for start, end, replacement in sorted(replacements, key=lambda item: (item[0], item[1]), reverse=True):
        merged = merged[:start] + replacement + merged[end:]

    return AnchorMergeResult(
        merged_text=merged,
        diff_ranges=tuple(diff_ranges),
        diff_points=tuple(sorted(diff_points)),
        applied_ops=tuple(applied_ops),
    )
