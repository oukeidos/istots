from __future__ import annotations

from pathlib import Path

from istots.pgs_engine import parser


def _make_pcs_payload(
    *,
    composition_state: int,
    palette_id: int = 0,
    composition_number: int = 1,
    object_id: int | None = 1,
) -> bytes:
    payload = bytearray()
    payload.extend((1920).to_bytes(2, byteorder="big"))
    payload.extend((1080).to_bytes(2, byteorder="big"))
    payload.append(0x10)
    payload.extend(composition_number.to_bytes(2, byteorder="big"))
    payload.append(composition_state)
    payload.append(0x00)
    payload.append(palette_id)
    if object_id is None:
        payload.append(0)
        return bytes(payload)

    payload.append(1)
    payload.extend(object_id.to_bytes(2, byteorder="big"))
    payload.append(0)
    payload.append(0)
    payload.extend((10).to_bytes(2, byteorder="big"))
    payload.extend((20).to_bytes(2, byteorder="big"))
    return bytes(payload)


def test_parse_display_set_decodes_complete_normal_state_from_cached_objects() -> None:
    engine = parser.PgsEngine(Path("dummy.sup"))
    engine._palettes = {
        0: {
            1: parser.PaletteEntry(y=255, cr=128, cb=128, alpha=255),
        }
    }
    engine._palette_versions = {0: 1}
    engine._objects = {
        1: parser.ObjectBuffer(
            object_id=1,
            width=2,
            height=2,
            complete=True,
            decoded_indices=[bytearray([1, 1]), bytearray([1, 1])],
        )
    }

    packets = [
        parser._Packet(
            pts=100,
            dts=100,
            segment_type=parser.SEGMENT_PCS,
            payload=_make_pcs_payload(composition_state=parser.COMPOSITION_STATE_NORMAL),
        ),
        parser._Packet(pts=100, dts=100, segment_type=parser.SEGMENT_END, payload=b""),
    ]

    row = engine._parse_display_set(0, packets)

    assert row.pcs is not None
    assert row.pcs.composition_state == parser.COMPOSITION_STATE_NORMAL
    assert row.complete is True
    assert row.decoded_pixels is not None
    assert len(row.decoded_windows) == 1
    assert len(row.decoded_pixels) == 2
    assert len(row.decoded_pixels[0]) == 2


def test_parse_display_set_marks_blank_rows_complete() -> None:
    engine = parser.PgsEngine(Path("dummy.sup"))
    packets = [
        parser._Packet(
            pts=200,
            dts=200,
            segment_type=parser.SEGMENT_PCS,
            payload=_make_pcs_payload(
                composition_state=parser.COMPOSITION_STATE_NORMAL,
                object_id=None,
            ),
        ),
        parser._Packet(pts=200, dts=200, segment_type=parser.SEGMENT_END, payload=b""),
    ]

    row = engine._parse_display_set(0, packets)

    assert row.pcs is not None
    assert row.pcs.composition_state == parser.COMPOSITION_STATE_NORMAL
    assert row.complete is True
    assert row.decoded_pixels is None
    assert row.decoded_windows == ()


def test_parse_display_set_composes_all_decoded_windows(monkeypatch) -> None:
    engine = parser.PgsEngine(Path("dummy.sup"))

    packets = [
        parser._Packet(
            pts=300,
            dts=300,
            segment_type=parser.SEGMENT_PCS,
            payload=_make_pcs_payload(composition_state=parser.COMPOSITION_STATE_NORMAL),
        ),
        parser._Packet(pts=300, dts=300, segment_type=parser.SEGMENT_END, payload=b""),
    ]

    monkeypatch.setattr(
        parser,
        "_decode_display_set_windows_cached",
        lambda **kwargs: (
            parser.DecodedWindow(
                window_id=0,
                left=0,
                top=0,
                right=1,
                bottom=1,
                object_ids=(1,),
                pixels=[[0, 0], [0, 0]],
            ),
            parser.DecodedWindow(
                window_id=1,
                left=3,
                top=1,
                right=4,
                bottom=1,
                object_ids=(2,),
                pixels=[[128, 128]],
            ),
        ),
    )

    row = engine._parse_display_set(0, packets)

    assert row.complete is True
    assert row.decoded_pixels == [
        [0, 0, 255, 255, 255],
        [0, 0, 255, 128, 128],
    ]
    assert len(row.decoded_windows) == 2


def test_parse_display_set_can_skip_full_surface_composition(monkeypatch) -> None:
    engine = parser.PgsEngine(Path("dummy.sup"))

    packets = [
        parser._Packet(
            pts=300,
            dts=300,
            segment_type=parser.SEGMENT_PCS,
            payload=_make_pcs_payload(composition_state=parser.COMPOSITION_STATE_NORMAL),
        ),
        parser._Packet(pts=300, dts=300, segment_type=parser.SEGMENT_END, payload=b""),
    ]

    monkeypatch.setattr(
        parser,
        "_decode_display_set_windows_cached",
        lambda **kwargs: (
            parser.DecodedWindow(
                window_id=0,
                left=0,
                top=0,
                right=1,
                bottom=1,
                object_ids=(1,),
                pixels=[[0, 0], [0, 0]],
            ),
            parser.DecodedWindow(
                window_id=1,
                left=3,
                top=1,
                right=4,
                bottom=1,
                object_ids=(2,),
                pixels=[[128, 128]],
            ),
        ),
    )
    monkeypatch.setattr(
        parser,
        "_compose_decoded_windows",
        lambda decoded_windows: (_ for _ in ()).throw(AssertionError("compose should be skipped")),
    )

    row = engine._parse_display_set(0, packets, include_decoded_pixels=False)

    assert row.complete is True
    assert row.decoded_pixels is None
    assert len(row.decoded_windows) == 2
