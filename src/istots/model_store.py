from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_GGUF_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5-GGUF"
DEFAULT_GGUF_FILENAME = "PaddleOCR-VL-1.5.gguf"
DEFAULT_GGUF_MMPROJ_FILENAME = "PaddleOCR-VL-1.5-mmproj.gguf"
DEFAULT_QWEN_CORRECTOR_MODEL_ID = "unsloth/Qwen3.5-35B-A3B-GGUF"
DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME = "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME = "mmproj-BF16.gguf"
MANAGED_SETUP_TARGET_SENTINEL = ".istots-managed-model-target"


@dataclass(frozen=True)
class SetupArtifacts:
    hf_model_dir: Path
    gguf_model_dir: Path
    gguf_model_path: Path
    gguf_mmproj_path: Path
    gguf_mmproj_minpix32768_path: Path
    qwen_corrector_dir: Path | None = None
    qwen_corrector_model_path: Path | None = None
    qwen_corrector_mmproj_path: Path | None = None


def default_models_dir() -> Path:
    configured = os.environ.get("ISTOTS_MODELS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "istots" / "models").resolve()


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def managed_setup_target_marker_path(target: Path) -> Path:
    return target / MANAGED_SETUP_TARGET_SENTINEL


def _validate_setup_model_id(model_id: str) -> None:
    direct = Path(model_id).expanduser()
    if direct.exists():
        raise RuntimeError(
            "setup model_id must be a Hugging Face repo ID, not an existing local path: "
            f"{direct.resolve()}"
        )


def resolve_setup_download_path(model_id: str, models_dir: Path | None = None) -> Path:
    _validate_setup_model_id(model_id)
    root = (models_dir or default_models_dir()).expanduser().resolve()
    return (root / model_dir_name(model_id)).resolve()


def _ensure_safe_force_delete_target(target: Path) -> None:
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError(f"refusing to delete non-directory setup target: {target}")
    marker = managed_setup_target_marker_path(target)
    if not marker.exists():
        raise RuntimeError(
            "refusing to delete existing setup target that is not marked as istots-managed: "
            f"{target}. Rerun `istots setup` without `--force` to initialize this target first, "
            "or remove it manually if the path is unintended."
        )


def _mark_setup_target_managed(target: Path, *, model_id: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    managed_setup_target_marker_path(target).write_text(
        "managed_by=istots_setup\n"
        f"repo_id={model_id}\n",
        encoding="utf-8",
    )


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
    target = resolve_setup_download_path(model_id=model_id, models_dir=models_dir)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for `istots setup`.") from exc

    target.parent.mkdir(parents=True, exist_ok=True)

    if force and target.exists():
        _ensure_safe_force_delete_target(target)
        shutil.rmtree(target)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        local_files_only=False,
    )
    _mark_setup_target_managed(target, model_id=model_id)
    return target


def _download_snapshot_files(
    *,
    model_id: str,
    required_filenames: tuple[str, ...],
    models_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, tuple[Path, ...]]:
    target = resolve_setup_download_path(model_id=model_id, models_dir=models_dir)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for `istots setup`.") from exc

    target.parent.mkdir(parents=True, exist_ok=True)

    if force and target.exists():
        _ensure_safe_force_delete_target(target)
        shutil.rmtree(target)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        local_files_only=False,
        allow_patterns=[
            *required_filenames,
            "LICENSE*",
            "README*",
            "*.json",
        ],
    )

    resolved_paths: list[Path] = []
    for filename in required_filenames:
        path = target / filename
        if not path.exists():
            raise RuntimeError(f"required file missing after setup: {path}")
        resolved_paths.append(path.resolve())
    _mark_setup_target_managed(target, model_id=model_id)
    return target.resolve(), tuple(resolved_paths)


def download_gguf_runtime_assets(
    *,
    model_id: str = DEFAULT_GGUF_MODEL_ID,
    models_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    target, paths = _download_snapshot_files(
        model_id=model_id,
        required_filenames=(
            DEFAULT_GGUF_FILENAME,
            DEFAULT_GGUF_MMPROJ_FILENAME,
        ),
        models_dir=models_dir,
        force=force,
    )
    model_path, mmproj_path = paths
    return target, model_path, mmproj_path


def download_qwen_corrector_assets(
    *,
    model_id: str = DEFAULT_QWEN_CORRECTOR_MODEL_ID,
    model_filename: str = DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME,
    mmproj_filename: str = DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME,
    models_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    target, paths = _download_snapshot_files(
        model_id=model_id,
        required_filenames=(model_filename, mmproj_filename),
        models_dir=models_dir,
        force=force,
    )
    model_path, mmproj_path = paths
    return target, model_path, mmproj_path


def ensure_local_qwen_corrector_assets(
    *,
    model_id: str = DEFAULT_QWEN_CORRECTOR_MODEL_ID,
    model_filename: str = DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME,
    mmproj_filename: str = DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME,
    models_dir: Path | None = None,
) -> tuple[Path, Path]:
    target = resolve_local_model_path(model_id=model_id, models_dir=models_dir)
    model_path = (target / model_filename).resolve()
    mmproj_path = (target / mmproj_filename).resolve()
    missing_paths = [path for path in (model_path, mmproj_path) if not path.exists()]
    if missing_paths:
        raise RuntimeError(
            "Local Qwen corrector assets are not available. "
            "Run `istots setup --with-qwen-corrector` or pass explicit "
            "`--corrector-model-path` and `--corrector-mmproj-path` values."
        )
    return model_path, mmproj_path


def setup_default_runtime_assets(
    *,
    hf_model_id: str = DEFAULT_MODEL_ID,
    gguf_model_id: str = DEFAULT_GGUF_MODEL_ID,
    with_qwen_corrector: bool = False,
    qwen_corrector_model_id: str = DEFAULT_QWEN_CORRECTOR_MODEL_ID,
    qwen_corrector_model_filename: str = DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME,
    qwen_corrector_mmproj_filename: str = DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME,
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
    qwen_corrector_dir: Path | None = None
    qwen_corrector_model_path: Path | None = None
    qwen_corrector_mmproj_path: Path | None = None
    if with_qwen_corrector:
        (
            qwen_corrector_dir,
            qwen_corrector_model_path,
            qwen_corrector_mmproj_path,
        ) = download_qwen_corrector_assets(
            model_id=qwen_corrector_model_id,
            model_filename=qwen_corrector_model_filename,
            mmproj_filename=qwen_corrector_mmproj_filename,
            models_dir=models_dir,
            force=force,
        )
    return SetupArtifacts(
        hf_model_dir=hf_model_dir,
        gguf_model_dir=gguf_model_dir,
        gguf_model_path=gguf_model_path,
        gguf_mmproj_path=gguf_mmproj_path,
        gguf_mmproj_minpix32768_path=derived_mmproj_path,
        qwen_corrector_dir=qwen_corrector_dir,
        qwen_corrector_model_path=qwen_corrector_model_path,
        qwen_corrector_mmproj_path=qwen_corrector_mmproj_path,
    )
