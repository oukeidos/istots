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
    raw_index: int
    pcs: _FakePcs | None
    complete: bool
    decoded_pixels: list[list[int]] | None
    decoded_windows: tuple[object, ...] = ()


@dataclass
class _FakeWindow:
    window_id: int
    left: int
    top: int
    right: int
    bottom: int
    pixels: list[list[int]]


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
            include_decoded_pixels: bool = True,
        ) -> list[_FakeDisplaySet]:
            captured["max_display_sets"] = max_display_sets
            captured["predecode_workers"] = predecode_workers
            captured["include_decoded_pixels"] = include_decoded_pixels
            return [
                _FakeDisplaySet(
                    raw_index=0,
                    pcs=_FakePcs(presentation_timestamp=100, composition_state=0x80),
                    complete=True,
                    decoded_pixels=[[1, 2], [3, 4]],
                ),
                _FakeDisplaySet(
                    raw_index=1,
                    pcs=_FakePcs(presentation_timestamp=120, composition_state=0x00),
                    complete=False,
                    decoded_pixels=None,
                ),
                _FakeDisplaySet(
                    raw_index=2,
                    pcs=_FakePcs(presentation_timestamp=200, composition_state=0x80),
                    complete=True,
                    decoded_pixels=[[1, 2], [3, 4]],
                ),
                _FakeDisplaySet(
                    raw_index=3,
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
    assert captured["include_decoded_pixels"] is True
    assert totals == [2]

    assert len(frames) == 2
    assert frames[0].raw_index == 0
    assert int(frames[0].start.total_seconds() * 1000) == 1
    assert int(frames[0].end.total_seconds() * 1000) == 3
    assert frames[0].image.size == (2, 2)
    assert frames[0].end > frames[0].start
    assert frames[1].raw_index == 3


def test_iter_sup_frames_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sup"
    try:
        next(iter(sup_reader.iter_sup_frames(missing)))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_iter_sup_window_frames_passes_cancel_callback_to_engine(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    seen_cancel_callbacks: list[object] = []

    class FakeEngine:
        def __init__(self, path: Path) -> None:
            self.path = path

        def parse_display_sets(
            self,
            max_display_sets: int | None = None,
            predecode_workers: int = 0,
            include_decoded_pixels: bool = True,
            cancel_callback=None,
        ) -> list[_FakeDisplaySet]:
            del max_display_sets, predecode_workers, include_decoded_pixels
            seen_cancel_callbacks.append(cancel_callback)
            return []

    monkeypatch.setattr(sup_reader, "PgsEngine", FakeEngine)

    frames = list(sup_reader.iter_sup_window_frames(input_sup, cancel_callback=lambda: None))

    assert frames == []
    assert len(seen_cancel_callbacks) == 1
    assert callable(seen_cancel_callbacks[0])


def test_iter_sup_window_frames_preserves_tracks_and_dedupes_per_window(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")

    captured: dict[str, object] = {}
    window_a = _FakeWindow(window_id=0, left=10, top=20, right=11, bottom=21, pixels=[[1, 1], [1, 1]])
    window_b = _FakeWindow(window_id=1, left=100, top=30, right=101, bottom=31, pixels=[[2, 2], [2, 2]])
    window_c = _FakeWindow(window_id=1, left=100, top=30, right=101, bottom=31, pixels=[[3, 3], [3, 3]])

    class FakeEngine:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        def parse_display_sets(
            self,
            max_display_sets: int | None = None,
            predecode_workers: int = 0,
            include_decoded_pixels: bool = True,
        ) -> list[_FakeDisplaySet]:
            captured["max_display_sets"] = max_display_sets
            captured["predecode_workers"] = predecode_workers
            captured["include_decoded_pixels"] = include_decoded_pixels
            return [
                _FakeDisplaySet(
                    raw_index=0,
                    pcs=_FakePcs(presentation_timestamp=100, composition_state=0x80),
                    complete=True,
                    decoded_pixels=window_a.pixels,
                    decoded_windows=(window_a, window_b),
                ),
                _FakeDisplaySet(
                    raw_index=1,
                    pcs=_FakePcs(presentation_timestamp=120, composition_state=0x00),
                    complete=False,
                    decoded_pixels=None,
                ),
                _FakeDisplaySet(
                    raw_index=2,
                    pcs=_FakePcs(presentation_timestamp=200, composition_state=0x80),
                    complete=True,
                    decoded_pixels=window_a.pixels,
                    decoded_windows=(window_a, window_c),
                ),
                _FakeDisplaySet(
                    raw_index=3,
                    pcs=_FakePcs(presentation_timestamp=220, composition_state=0x00),
                    complete=True,
                    decoded_pixels=None,
                ),
            ]

    monkeypatch.setattr(sup_reader, "PgsEngine", FakeEngine)

    totals: list[int] = []
    frames = list(sup_reader.iter_sup_window_frames(input_sup, max_items=12, on_total=totals.append))

    assert captured["path"] == input_sup
    assert captured["max_display_sets"] is None
    assert captured["predecode_workers"] == -1
    assert captured["include_decoded_pixels"] is False
    assert totals == [3]

    assert len(frames) == 3
    assert frames[0].window_id == 0
    assert frames[0].raw_index == 0
    assert int(frames[0].start.total_seconds() * 1000) == 1
    assert int(frames[0].end.total_seconds() * 1000) == 2
    assert frames[0].image.size == (2, 2)

    assert frames[1].window_id == 1
    assert frames[1].raw_index == 0
    assert int(frames[1].start.total_seconds() * 1000) == 1
    assert int(frames[1].end.total_seconds() * 1000) == 2

    assert frames[2].window_id == 1
    assert frames[2].raw_index == 2
    assert int(frames[2].start.total_seconds() * 1000) == 2
    assert int(frames[2].end.total_seconds() * 1000) == 3


def test_iter_sup_frames_uses_complete_normal_updates_as_continuity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")

    class FakeEngine:
        def __init__(self, path: Path) -> None:
            self.path = path

        def parse_display_sets(
            self,
            max_display_sets: int | None = None,
            predecode_workers: int = 0,
            include_decoded_pixels: bool = True,
        ) -> list[_FakeDisplaySet]:
            return [
                _FakeDisplaySet(
                    raw_index=0,
                    pcs=_FakePcs(presentation_timestamp=100, composition_state=0x80),
                    complete=True,
                    decoded_pixels=[[1, 2], [3, 4]],
                ),
                _FakeDisplaySet(
                    raw_index=1,
                    pcs=_FakePcs(presentation_timestamp=200, composition_state=0x00),
                    complete=True,
                    decoded_pixels=[[1, 2], [3, 4]],
                ),
                _FakeDisplaySet(
                    raw_index=2,
                    pcs=_FakePcs(presentation_timestamp=300, composition_state=0x00),
                    complete=True,
                    decoded_pixels=None,
                ),
            ]

    monkeypatch.setattr(sup_reader, "PgsEngine", FakeEngine)

    frames = list(sup_reader.iter_sup_frames(input_sup))

    assert len(frames) == 1
    assert frames[0].raw_index == 0
    assert int(frames[0].start.total_seconds() * 1000) == 1
    assert int(frames[0].end.total_seconds() * 1000) == 3
