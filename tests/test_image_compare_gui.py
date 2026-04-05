from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image


def _load_module():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    try:
        from tools.image_compare_gui import compare_images
    finally:
        sys.path.pop(0)
    return compare_images


def test_scan_image_files_and_entry_order(tmp_path: Path) -> None:
    compare_images = _load_module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    Image.new("RGB", (10, 10), "white").save(left / "0001.png")
    Image.new("RGB", (10, 10), "black").save(right / "0001.png")
    Image.new("RGB", (10, 10), "red").save(left / "only_left.png")
    (left / "ignore.txt").write_text("x", encoding="utf-8")

    panels = [
        compare_images.PanelSpec("left", left, compare_images.scan_image_files(left)),
        compare_images.PanelSpec("right", right, compare_images.scan_image_files(right)),
    ]

    assert sorted(panels[0].files) == ["0001.png", "only_left.png"]
    assert sorted(panels[1].files) == ["0001.png"]
    assert compare_images.build_entry_order(panels, intersection_only=False) == [
        "0001.png",
        "only_left.png",
    ]
    assert compare_images.build_entry_order(panels, intersection_only=True) == ["0001.png"]


def test_masking_only_filters_entries_via_manifest(tmp_path: Path) -> None:
    compare_images = _load_module()
    root = tmp_path / "debug"
    original = root / "original"
    masked = root / "masked"
    original.mkdir(parents=True)
    masked.mkdir(parents=True)

    Image.new("RGB", (10, 10), "white").save(original / "000001_w0.png")
    Image.new("RGB", (10, 10), "white").save(original / "000001_w1.png")
    Image.new("RGB", (10, 10), "white").save(original / "000002_w0.png")
    Image.new("RGB", (10, 10), "white").save(masked / "000001_w0.png")
    Image.new("RGB", (10, 10), "white").save(masked / "000001_w1.png")
    Image.new("RGB", (10, 10), "white").save(masked / "000002_w0.png")

    (root / "manifest.jsonl").write_text(
        "\n".join(
            [
                '{"frame_index":1,"segment_index":1,"window_id":0,"bbox":[0,0,9,9],"original":"original/000001_w0.png","masked":"masked/000001_w0.png","selected_count":2,"masked_pixel_count":40}',
                '{"frame_index":1,"segment_index":2,"window_id":1,"bbox":[10,0,19,9],"original":"original/000001_w1.png","masked":"masked/000001_w1.png","selected_count":0,"masked_pixel_count":0}',
                '{"frame_index":2,"segment_index":1,"window_id":0,"bbox":[0,10,9,19],"original":"original/000002_w0.png","masked":"masked/000002_w0.png","selected_count":0,"masked_pixel_count":0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    panels = [
        compare_images.PanelSpec("original", original, compare_images.scan_image_files(original)),
        compare_images.PanelSpec("masked", masked, compare_images.scan_image_files(masked)),
    ]

    manifest_path = compare_images.resolve_manifest_path(panels, None)
    manifest_index = compare_images.load_manifest_index(manifest_path)
    entries = compare_images.build_entry_order(
        panels,
        intersection_only=False,
        allowed_entries=manifest_index.masked_entries,
        manifest_order=manifest_index.ordered_entries,
    )

    assert manifest_path == root / "manifest.jsonl"
    assert manifest_index.masked_entries == {"000001_w0.png", "000001_w1.png"}
    assert entries == ["000001_w0.png", "000001_w1.png"]
    assert manifest_index.metadata_by_entry["000001_w1.png"].window_id == 1
    assert manifest_index.metadata_by_entry["000001_w1.png"].bbox == (10, 0, 19, 9)


def test_find_entry_index_accepts_segment_number_and_filename() -> None:
    compare_images = _load_module()
    entries = ["000001_w1.png", "nested/000037_w0.png", "000120_w0.png"]

    assert compare_images.find_entry_index(entries, "37") == 1
    assert compare_images.find_entry_index(entries, "000037") == 1
    assert compare_images.find_entry_index(entries, "000037_w0.png") == 1
    assert compare_images.find_entry_index(entries, "nested/000037_w0.png") == 1


def test_build_entry_order_respects_manifest_order(tmp_path: Path) -> None:
    compare_images = _load_module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    for name in ("000002_w0.png", "000001_w1.png", "000001_w0.png"):
        Image.new("RGB", (10, 10), "white").save(left / name)
        Image.new("RGB", (10, 10), "white").save(right / name)

    panels = [
        compare_images.PanelSpec("left", left, compare_images.scan_image_files(left)),
        compare_images.PanelSpec("right", right, compare_images.scan_image_files(right)),
    ]
    manifest_order = ("000001_w0.png", "000001_w1.png", "000002_w0.png")

    entries = compare_images.build_entry_order(
        panels,
        intersection_only=False,
        manifest_order=manifest_order,
    )

    assert entries == ["000001_w0.png", "000001_w1.png", "000002_w0.png"]
