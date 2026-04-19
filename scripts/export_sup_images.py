from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from istots.sup_reader import iter_sup_window_frames

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deduplicated subtitle window images from SUP. "
            "Uses istots Python parser/policy path (same OCR-input basis as conversion)."
        ),
    )
    parser.add_argument("input_sup", type=Path, help="Input .sup file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Export only first N deduplicated subtitle window inputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove output directory first if it already exists",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )
    return parser.parse_args()


def timedelta_to_ms(value) -> int:
    return int(value.total_seconds() * 1000)


def ensure_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise RuntimeError(
                f"output directory already exists: {output_dir} "
                "(use --force to overwrite)"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)


def export_images(input_sup: Path, output_dir: Path, max_items: int | None) -> int:
    total_hint = 0

    def on_total(count: int) -> None:
        nonlocal total_hint
        total_hint = count
        logging.info("deduplicated subtitle windows detected: %d", count)

    image_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"

    written = 0
    frame_index = 0
    segment_index = 0
    current_group: tuple[int, int] | None = None
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, frame in enumerate(
            iter_sup_window_frames(
                input_sup=input_sup,
                max_items=max_items,
                on_total=on_total,
            ),
            start=1,
        ):
            start_ms = timedelta_to_ms(frame.start)
            end_ms = timedelta_to_ms(frame.end)
            if end_ms <= start_ms:
                end_ms = start_ms + 1

            group_key = (frame.raw_index, start_ms)
            if group_key != current_group:
                current_group = group_key
                frame_index += 1
                segment_index = 0
            segment_index += 1

            filename = f"{frame_index:06d}_w{frame.window_id}.png"
            image_path = image_dir / filename
            frame.image.save(image_path, format="PNG")

            row = {
                "index": index,
                "frame_index": frame_index,
                "segment_index": segment_index,
                "raw_index": frame.raw_index,
                "window_id": frame.window_id,
                "bbox": [frame.left, frame.top, frame.right, frame.bottom],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "image": f"images/{filename}",
            }
            manifest.write(json.dumps(row, ensure_ascii=True) + "\n")
            written = index

            if written % 100 == 0:
                if total_hint > 0:
                    logging.info("exported %d/%d", written, total_hint)
                else:
                    logging.info("exported %d", written)

    return written


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.max_items is not None and args.max_items <= 0:
        raise RuntimeError("--max-items must be a positive integer")

    input_sup = args.input_sup.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_sup.exists():
        raise FileNotFoundError(f"input not found: {input_sup}")

    ensure_output_dir(output_dir, force=args.force)
    logging.info("input: %s", input_sup)
    logging.info("output: %s", output_dir)

    try:
        written = export_images(
            input_sup=input_sup,
            output_dir=output_dir,
            max_items=args.max_items,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to export deduplicated SUP images."
        ) from exc

    logging.info("done: exported %d images", written)
    logging.info("manifest: %s", output_dir / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
