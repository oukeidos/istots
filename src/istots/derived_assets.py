from __future__ import annotations

import os
from pathlib import Path

from istots.llama_mmproj import DEFAULT_MIN_PIXELS, default_materialized_mmproj_path

DERIVED_MMPROJ_ROOT_ENV = "ISTOTS_DERIVED_MMPROJ_DIR"


def default_derived_mmproj_root(models_dir: Path | None = None) -> Path:
    configured = os.environ.get(DERIVED_MMPROJ_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()

    models_root = _default_models_root(models_dir=models_dir)
    return (models_root.parent / "derived" / "mmproj").resolve()


def resolve_derived_mmproj_output_path(
    *,
    base_mmproj: Path,
    models_dir: Path | None = None,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    derived_root: Path | None = None,
) -> Path:
    root = (derived_root or default_derived_mmproj_root(models_dir=models_dir)).expanduser().resolve()
    filename = default_materialized_mmproj_path(base_mmproj, min_pixels).name
    return (root / base_mmproj.parent.name / filename).resolve()


def _default_models_root(models_dir: Path | None) -> Path:
    if models_dir is not None:
        return models_dir.expanduser().resolve()
    configured = os.environ.get("ISTOTS_MODELS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "istots" / "models").resolve()
