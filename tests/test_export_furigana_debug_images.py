from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "export_furigana_debug_images.py"
    spec = importlib.util.spec_from_file_location("export_furigana_debug_images", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_debug_images_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_dir = tmp_path / "out"
    module.ensure_output_dir(output_dir, force=False)

    frame = SimpleNamespace(
        raw_index=7,
        window_id=3,
        left=10,
        top=20,
        right=21,
        bottom=27,
        start=SimpleNamespace(total_seconds=lambda: 1.0),
        end=SimpleNamespace(total_seconds=lambda: 2.0),
        image=Image.new("RGB", (12, 8), "white"),
    )

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](1)
        return iter([frame])

    fake_result = SimpleNamespace(
        image=Image.new("RGB", (12, 8), "black"),
        mask=Image.new("L", (12, 8), 255),
        orientation="horizontal",
        component_count=6,
        selected_count=2,
        masked_pixel_count=24,
        lines=(
            SimpleNamespace(left=0, top=0, right=11, bottom=2, role="main"),
            SimpleNamespace(left=0, top=4, right=11, bottom=7, role="furigana"),
        ),
    )

    monkeypatch.setattr(module, "iter_sup_window_frames", fake_iter_sup_window_frames)
    monkeypatch.setattr(module, "build_furigana_masks", lambda images: [fake_result for _ in images])

    written = module.export_debug_images(input_sup, output_dir, max_items=None)

    assert written == 1
    assert (output_dir / "original" / "000001_w3.png").exists()
    assert (output_dir / "masked" / "000001_w3.png").exists()
    assert (output_dir / "mask" / "000001_w3.png").exists()
    assert (output_dir / "lines" / "000001_w3.png").exists()

    rows = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    manifest_row = json.loads(rows[0])
    assert manifest_row["frame_index"] == 1
    assert manifest_row["segment_index"] == 1
    assert manifest_row["raw_index"] == 7
    assert manifest_row["window_id"] == 3
    assert manifest_row["bbox"] == [10, 20, 21, 27]
    assert manifest_row["orientation"] == "horizontal"
    assert manifest_row["selected_count"] == 2
    assert manifest_row["masked_pixel_count"] == 24
    assert manifest_row["main_line_count"] == 1
    assert manifest_row["furigana_line_count"] == 1


def test_draw_line_overlay_uses_blue_for_main_and_red_for_furigana() -> None:
    module = _load_script_module()
    image = Image.new("RGB", (12, 8), "white")
    lines = (
        SimpleNamespace(left=0, top=0, right=11, bottom=2, role="main"),
        SimpleNamespace(left=0, top=4, right=11, bottom=7, role="furigana"),
    )

    overlay = module.draw_line_overlay(image, lines)

    assert overlay.getpixel((0, 0)) == (0, 0, 255)
    assert overlay.getpixel((0, 4)) == (255, 0, 0)
