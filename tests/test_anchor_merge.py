from __future__ import annotations

from istots.anchor_merge import apply_union_anchor_merge, build_focus_context


def test_build_focus_context_tracks_replace_and_insert_anchors() -> None:
    rows = build_focus_context("ABC", "ABXC")

    assert len(rows) == 1
    assert rows[0].tag == "insert"
    assert rows[0].baseline_span == (2, 2)
    assert rows[0].option_span == (2, 3)
    assert rows[0].baseline_frag == ""
    assert rows[0].option_frag == "X"


def test_apply_union_anchor_merge_replaces_only_anchor_overlaps() -> None:
    rows = build_focus_context("ABC", "ADC")
    result = apply_union_anchor_merge("ABC", "AECZ", rows)

    assert result.merged_text == "AEC"
    assert len(result.applied_ops) == 1
    assert result.applied_ops[0].baseline_frag == "B"
    assert result.applied_ops[0].candidate_frag == "E"


def test_apply_union_anchor_merge_supports_anchor_point_insertions() -> None:
    rows = build_focus_context("ABC", "ABXC")
    result = apply_union_anchor_merge("ABC", "ABYCZ", rows)

    assert result.merged_text == "ABYC"
    assert len(result.applied_ops) == 1
    assert result.applied_ops[0].tag == "insert"
