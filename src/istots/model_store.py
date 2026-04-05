from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"


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
