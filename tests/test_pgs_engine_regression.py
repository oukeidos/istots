from __future__ import annotations

from pathlib import Path

import pytest

from istots.pgs_engine import PgsEngine
from istots.pgs_engine.parser import hash_gray_pixels


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sample_sup_path() -> Path:
    return (_repo_root().parent / "test" / "test.sup").resolve()


@pytest.mark.skipif(not _sample_sup_path().exists(), reason="sample SUP file not found")
def test_layer_a_parser_regression_subset() -> None:
    input_sup = _sample_sup_path()
    rows = PgsEngine(input_sup).parse_display_sets(max_display_sets=300, predecode_workers=-1)

    assert len(rows) == 300
    assert sum(1 for row in rows if row.pcs is not None) == 300
    assert sum(1 for row in rows if row.decoded_pixels is not None) == 248
    assert sum(1 for row in rows if row.pcs and row.pcs.composition_state in (0x80, 0x40)) == 248

    # Fixed checkpoints to catch parser/palette/RLE/composition regressions.
    checkpoints = {
        0: (739487, 0x80, True, 522, 64, 15700377827298176001),
        14: (1159907, 0x40, True, 522, 64, 15700377827298176001),
        15: (1268766, 0x00, False, None, None, None),
        16: (3622367, 0x80, True, 970, 160, 17945537943421376721),
        100: (7785276, 0x40, True, 646, 127, 16563544240394164936),
        150: (11629116, 0x40, True, 227, 64, 6168999088873797973),
        200: (13900135, 0x00, False, None, None, None),
        250: (17030763, 0x80, True, 654, 128, 14778950278019398575),
        299: (19493222, 0x80, True, 63, 563, 14208084348866343546),
    }

    for idx, expected in checkpoints.items():
        pts, comp_state, decoded_expected, width, height, expected_hash = expected
        row = rows[idx]
        assert row.pcs is not None
        assert row.pcs.presentation_timestamp == pts
        assert row.pcs.composition_state == comp_state
        assert (row.decoded_pixels is not None) == decoded_expected

        if decoded_expected:
            assert row.decoded_pixels is not None
            assert len(row.decoded_pixels[0]) == width
            assert len(row.decoded_pixels) == height
            assert hash_gray_pixels(row.decoded_pixels) == expected_hash
