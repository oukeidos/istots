from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_GGUF_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5-GGUF"
DEFAULT_GGUF_FILENAME = "PaddleOCR-VL-1.5.gguf"
DEFAULT_GGUF_MMPROJ_FILENAME = "PaddleOCR-VL-1.5-mmproj.gguf"


@dataclass(frozen=True)
class SetupArtifacts:
    hf_model_dir: Path
    gguf_model_dir: Path
    gguf_model_path: Path
    gguf_mmproj_path: Path
    gguf_mmproj_minpix32768_path: Path


def default_models_dir() -> Path:
    configured = os.environ.get("ISTOTS_MODELS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "istots" / "models").resolve()


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def resolve_local_model_path(model_id: str, models_dir: Path | None = None) -> Path:
    direct = Path(model_id).expanduser()
    if direct.exists():
        return direct.resolve()

    root = (models_dir or default_models_dir()).expanduser().resolve()
    return root / model_dir_name(model_id)


def ensure_local_model(model_id: str, models_dir: Path | None = None) -> Path:
    path = resolve_local_model_path(model_id=model_id, models_dir=models_dir)
    if not path.exists() or not (path / "config.json").exists():
        raise RuntimeError(
            "Model is not available locally. "
            f"Expected at: {path}. Run `istots setup --model-id {model_id}` first."
        )
    return path


def download_model(model_id: str, models_dir: Path | None = None, force: bool = False) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for `istots setup`.") from exc

    target = resolve_local_model_path(model_id=model_id, models_dir=models_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    if force and target.exists():
        shutil.rmtree(target)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        local_files_only=False,
    )
    return target


def download_gguf_runtime_assets(
    *,
    model_id: str = DEFAULT_GGUF_MODEL_ID,
    models_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for `istots setup`.") from exc

    target = resolve_local_model_path(model_id=model_id, models_dir=models_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    if force and target.exists():
        shutil.rmtree(target)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        local_files_only=False,
        allow_patterns=[
            DEFAULT_GGUF_FILENAME,
            DEFAULT_GGUF_MMPROJ_FILENAME,
            "LICENSE*",
            "README*",
            "*.json",
        ],
    )

    model_path = target / DEFAULT_GGUF_FILENAME
    mmproj_path = target / DEFAULT_GGUF_MMPROJ_FILENAME
    if not model_path.exists():
        raise RuntimeError(f"GGUF model file missing after setup: {model_path}")
    if not mmproj_path.exists():
        raise RuntimeError(f"GGUF mmproj file missing after setup: {mmproj_path}")
    return target, model_path, mmproj_path


def setup_default_runtime_assets(
    *,
    hf_model_id: str = DEFAULT_MODEL_ID,
    gguf_model_id: str = DEFAULT_GGUF_MODEL_ID,
    models_dir: Path | None = None,
    force: bool = False,
    support_dir: Path | None = None,
    gguf_py_base_url: str | None = None,
    gguf_source_mode: str = "auto",
    min_pixels: int = 32768,
) -> SetupArtifacts:
    from istots.llama_mmproj import materialize_mmproj

    hf_model_dir = download_model(
        model_id=hf_model_id,
        models_dir=models_dir,
        force=force,
    )
    gguf_model_dir, gguf_model_path, gguf_mmproj_path = download_gguf_runtime_assets(
        model_id=gguf_model_id,
        models_dir=models_dir,
        force=force,
    )
    derived_mmproj_path = materialize_mmproj(
        base_mmproj=gguf_mmproj_path,
        min_pixels=min_pixels,
        support_dir=support_dir,
        gguf_py_base_url=gguf_py_base_url,
        gguf_source_mode=gguf_source_mode,
        force=force,
    )
    return SetupArtifacts(
        hf_model_dir=hf_model_dir,
        gguf_model_dir=gguf_model_dir,
        gguf_model_path=gguf_model_path,
        gguf_mmproj_path=gguf_mmproj_path,
        gguf_mmproj_minpix32768_path=derived_mmproj_path,
    )
