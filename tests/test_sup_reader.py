from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from istots import sup_reader


@dataclass
class _FakePcs:
    presentation_timestamp: int
    composition_state: int


@dataclass
class _FakeDisplaySet:
    pcs: _FakePcs | None
    complete: bool
    decoded_pixels: list[list[int]] | None


def test_iter_sup_frames_uses_python_engine_and_reports_total(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")

    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        def parse_display_sets(
            self,
            max_display_sets: int | None = None,
            predecode_workers: int = 0,
        ) -> list[_FakeDisplaySet]:
            captured["max_display_sets"] = max_display_sets
            captured["predecode_workers"] = predecode_workers
            return [
                _FakeDisplaySet(
                    pcs=_FakePcs(presentation_timestamp=100, composition_state=0x80),
                    complete=True,
                    decoded_pixels=[[1, 2], [3, 4]],
                ),
                _FakeDisplaySet(
                    pcs=_FakePcs(presentation_timestamp=120, composition_state=0x00),
                    complete=False,
                    decoded_pixels=None,
                ),
                _FakeDisplaySet(
                    pcs=_FakePcs(presentation_timestamp=200, composition_state=0x80),
                    complete=True,
                    decoded_pixels=[[1, 2], [3, 4]],
                ),
                _FakeDisplaySet(
                    pcs=_FakePcs(presentation_timestamp=260, composition_state=0x80),
                    complete=True,
                    decoded_pixels=[[5, 6], [7, 8]],
                ),
            ]

    monkeypatch.setattr(sup_reader, "PgsEngine", FakeEngine)

    totals: list[int] = []
    frames = list(sup_reader.iter_sup_frames(input_sup, max_items=12, on_total=totals.append))

    assert captured["path"] == input_sup
    assert captured["max_display_sets"] is None
    assert captured["predecode_workers"] == -1
    assert totals == [1]

    assert len(frames) == 1
    assert frames[0].raw_index == 0
    assert int(frames[0].start.total_seconds() * 1000) == 1
    assert int(frames[0].end.total_seconds() * 1000) == 3
    assert frames[0].image.size == (2, 2)
    assert frames[0].end > frames[0].start


def test_iter_sup_frames_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sup"
    try:
        next(iter(sup_reader.iter_sup_frames(missing)))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
