from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "export_sup_images.py"
    spec = importlib.util.spec_from_file_location("export_sup_images", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_images_writes_window_aware_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    input_sup = tmp_path / "input.sup"
    input_sup.write_bytes(b"PG")
    output_dir = tmp_path / "out"
    module.ensure_output_dir(output_dir, force=False)

    frames = [
        SimpleNamespace(
            raw_index=7,
            window_id=0,
            left=10,
            top=20,
            right=21,
            bottom=27,
            start=SimpleNamespace(total_seconds=lambda: 1.0),
            end=SimpleNamespace(total_seconds=lambda: 2.0),
            image=Image.new("RGB", (12, 8), "white"),
        ),
        SimpleNamespace(
            raw_index=7,
            window_id=1,
            left=100,
            top=200,
            right=111,
            bottom=207,
            start=SimpleNamespace(total_seconds=lambda: 1.0),
            end=SimpleNamespace(total_seconds=lambda: 2.0),
            image=Image.new("RGB", (12, 8), "gray"),
        ),
        SimpleNamespace(
            raw_index=8,
            window_id=0,
            left=30,
            top=40,
            right=41,
            bottom=47,
            start=SimpleNamespace(total_seconds=lambda: 3.0),
            end=SimpleNamespace(total_seconds=lambda: 4.0),
            image=Image.new("RGB", (12, 8), "black"),
        ),
    ]

    def fake_iter_sup_window_frames(*args, **kwargs):
        if kwargs.get("on_total") is not None:
            kwargs["on_total"](3)
        return iter(frames)

    monkeypatch.setattr(module, "iter_sup_window_frames", fake_iter_sup_window_frames)

    written = module.export_images(input_sup, output_dir, max_items=None)

    assert written == 3
    assert (output_dir / "images" / "000001_w0.png").exists()
    assert (output_dir / "images" / "000001_w1.png").exists()
    assert (output_dir / "images" / "000002_w0.png").exists()

    rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert rows == [
        {
            "index": 1,
            "frame_index": 1,
            "segment_index": 1,
            "raw_index": 7,
            "window_id": 0,
            "bbox": [10, 20, 21, 27],
            "start_ms": 1000,
            "end_ms": 2000,
            "image": "images/000001_w0.png",
        },
        {
            "index": 2,
            "frame_index": 1,
            "segment_index": 2,
            "raw_index": 7,
            "window_id": 1,
            "bbox": [100, 200, 111, 207],
            "start_ms": 1000,
            "end_ms": 2000,
            "image": "images/000001_w1.png",
        },
        {
            "index": 3,
            "frame_index": 2,
            "segment_index": 1,
            "raw_index": 8,
            "window_id": 0,
            "bbox": [30, 40, 41, 47],
            "start_ms": 3000,
            "end_ms": 4000,
            "image": "images/000002_w0.png",
        },
    ]
