from __future__ import annotations

from pathlib import Path

import pytest

from istots.pgs_engine import PgsEngine
from istots.pgs_engine.parser import hash_gray_pixels


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _smoke_sample_sup_path() -> Path:
    return (_repo_root().parent / "test" / "sample.sup").resolve()


@pytest.mark.skipif(not _smoke_sample_sup_path().exists(), reason="smoke sample SUP file not found")
def test_smoke_sample_sup_parser_contract() -> None:
    input_sup = _smoke_sample_sup_path()
    rows = PgsEngine(input_sup).parse_display_sets(predecode_workers=-1)

    assert len(rows) == 5
    assert sum(1 for row in rows if row.pcs is not None) == 5
    assert sum(1 for row in rows if row.decoded_pixels is not None) == 5

    checkpoints = {
        0: (0, 0x80, 713, 124, 1641419132810084962),
        1: (270000, 0x80, 129, 609, 10390759913425770849),
        2: (540000, 0x80, 882, 128, 16057723198783394134),
        3: (810000, 0x80, 65, 503, 12101918009803205878),
        4: (1080000, 0x80, 65, 503, 12101918009803205878),
    }

    for idx, expected in checkpoints.items():
        pts, comp_state, width, height, expected_hash = expected
        row = rows[idx]
        assert row.pcs is not None
        assert row.decoded_pixels is not None
        assert row.pcs.presentation_timestamp == pts
        assert row.pcs.composition_state == comp_state
        assert len(row.decoded_pixels[0]) == width
        assert len(row.decoded_pixels) == height
        assert hash_gray_pixels(row.decoded_pixels) == expected_hash
