from __future__ import annotations

from istots.pgs_engine import parser


def test_decode_display_set_windows_returns_each_window_separately() -> None:
    pcs = parser.PcsSegment(
        width=1920,
        height=1080,
        frame_rate=24,
        composition_number=1,
        composition_state=0x80,
        palette_update_flag=0,
        palette_id=0,
        presentation_timestamp=1234,
        objects=[
            parser.CompositionObject(object_id=1, window_id=0, x=10, y=20),
            parser.CompositionObject(object_id=2, window_id=1, x=100, y=30),
        ],
    )
    palettes = {
        0: {
            1: parser.PaletteEntry(y=255, cr=128, cb=128, alpha=255),
        }
    }
    objects = {
        1: parser.ObjectBuffer(
            object_id=1,
            width=2,
            height=2,
            complete=True,
            decoded_indices=[bytearray([1, 1]), bytearray([1, 1])],
        ),
        2: parser.ObjectBuffer(
            object_id=2,
            width=3,
            height=1,
            complete=True,
            decoded_indices=[bytearray([1, 1, 1])],
        ),
    }

    decoded = parser._decode_display_set_windows(
        pcs=pcs,
        palettes=palettes,
        palette_versions={0: 1},
        objects=objects,
        predecoded_objects=None,
    )

    assert [window.window_id for window in decoded] == [0, 1]

    first = decoded[0]
    assert first.object_ids == (1,)
    assert first.size == (2, 2)
    assert (first.left, first.top, first.right, first.bottom) == (10, 20, 11, 21)

    second = decoded[1]
    assert second.object_ids == (2,)
    assert second.size == (3, 1)
    assert (second.left, second.top, second.right, second.bottom) == (100, 30, 102, 30)


def test_compose_decoded_windows_returns_union_surface() -> None:
    decoded = parser._compose_decoded_windows(
        (
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
        )
    )

    assert decoded == [
        [0, 0, 255, 255, 255],
        [0, 0, 255, 128, 128],
    ]
