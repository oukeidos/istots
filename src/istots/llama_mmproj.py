from __future__ import annotations

import shutil
from pathlib import Path

from istots.gguf_support import load_known_good_gguf

MMPROJ_MIN_PIXELS_KEY = "clip.vision.image_min_pixels"
DEFAULT_MIN_PIXELS = 32768


def default_materialized_mmproj_path(base_mmproj: Path, min_pixels: int = DEFAULT_MIN_PIXELS) -> Path:
    suffix = base_mmproj.suffix
    stem = base_mmproj.name[: -len(suffix)] if suffix else base_mmproj.name
    return base_mmproj.with_name(f"{stem}.minpix{min_pixels}{suffix}")


def materialize_mmproj(
    *,
    base_mmproj: Path,
    output_path: Path | None = None,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    support_dir: Path | None = None,
    gguf_py_base_url: str | None = None,
    gguf_source_mode: str = "auto",
    force: bool = False,
) -> Path:
    base_mmproj = base_mmproj.expanduser().resolve()
    if not base_mmproj.exists():
        raise RuntimeError(f"base mmproj does not exist: {base_mmproj}")
    if min_pixels <= 0:
        raise RuntimeError("min_pixels must be positive")

    output = (output_path or default_materialized_mmproj_path(base_mmproj, min_pixels)).expanduser().resolve()
    if output == base_mmproj:
        raise RuntimeError("output path must differ from the base mmproj path")
    output.parent.mkdir(parents=True, exist_ok=True)

    gguf = load_known_good_gguf(
        source_mode=gguf_source_mode,
        support_dir=support_dir,
        base_url=gguf_py_base_url,
    )

    if output.exists() and not force:
        current_value = read_mmproj_min_pixels(output, gguf_module=gguf)
        if current_value == min_pixels:
            return output
        raise RuntimeError(
            f"output already exists with clip.vision.image_min_pixels={current_value}: {output}. "
            "Rerun with --force to overwrite."
        )

    temp_output = output.with_name(f".{output.name}.tmp")
    if temp_output.exists():
        temp_output.unlink()
    shutil.copy2(base_mmproj, temp_output)

    reader = gguf.GGUFReader(str(temp_output), "r+")
    field = reader.get_field(MMPROJ_MIN_PIXELS_KEY)
    if field is None:
        raise RuntimeError(f"metadata field missing: {MMPROJ_MIN_PIXELS_KEY}")
    handler = reader.gguf_scalar_to_np.get(field.types[0]) if field.types else None
    if handler is None:
        raise RuntimeError(
            f"unsupported GGUF field type for {MMPROJ_MIN_PIXELS_KEY}: {field.types}"
        )
    field.parts[field.data[0]][0] = handler(min_pixels)
    reader.data.flush()
    del reader

    written_value = read_mmproj_min_pixels(temp_output, gguf_module=gguf)
    if written_value != min_pixels:
        raise RuntimeError(
            "failed to materialize mmproj with requested min_pixels: "
            f"expected {min_pixels}, got {written_value}"
        )

    temp_output.replace(output)
    return output


def read_mmproj_min_pixels(
    path: Path,
    *,
    gguf_module=None,
    support_dir: Path | None = None,
    gguf_py_base_url: str | None = None,
    gguf_source_mode: str = "auto",
) -> int:
    path = path.expanduser().resolve()
    gguf = gguf_module
    if gguf is None:
        gguf = load_known_good_gguf(
            source_mode=gguf_source_mode,
            support_dir=support_dir,
            base_url=gguf_py_base_url,
        )

    reader = gguf.GGUFReader(str(path), "r")
    field = reader.get_field(MMPROJ_MIN_PIXELS_KEY)
    if field is None:
        raise RuntimeError(f"metadata field missing: {MMPROJ_MIN_PIXELS_KEY}")
    return int(field.parts[field.data[0]][0])
