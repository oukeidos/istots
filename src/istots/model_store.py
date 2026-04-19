from __future__ import annotations

import hashlib
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
    hf_model_dir: Path | None
    gguf_model_dir: Path
    gguf_model_path: Path
    gguf_mmproj_path: Path
    gguf_mmproj_minpix32768_path: Path
    qwen_corrector_dir: Path | None = None
    qwen_corrector_model_path: Path | None = None
    qwen_corrector_mmproj_path: Path | None = None


@dataclass(frozen=True)
class PinnedSnapshotBundle:
    repo_id: str
    revision: str
    file_hashes: dict[str, str]


DEFAULT_HF_MODEL_BUNDLE = PinnedSnapshotBundle(
    repo_id=DEFAULT_MODEL_ID,
    revision="6819afc8509ac9afa50e91b34627a7cf8f7900bb",
    file_hashes={
        ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
        "LICENSE": "b8c4d7deccd236af023af1c88c4d4e8f0fc2f41914e0fb23a3ec9678fb5a8456",
        "README.md": "94bbd60d9ed606d4f1f4e056bc2e56f5cbc59cadbd65bb3764022da8eef6bbe7",
        "added_tokens.json": "f59f889088e0fe21c523e7cf121bb6dca3b0bb148cb7159fbb4572c74dfc5644",
        "chat_template.jinja": "2f27812dab7f333e471884e0c803d807f11953d5453140dfb1aaba234f872bc8",
        "config.json": "ce7f4565f8b1db78532ad5d1b9ebe55c2139d49bd4cb04778b580a08a598f171",
        "configuration_paddleocr_vl.py": "753dd93654c3a9c8c85a3eaee1e3092dd12591b0f2dce0305e1abfb7a41ff160",
        "generation_config.json": "a6701d78ab3b4d972307cdec3b69d4c13f46e0d5140514f50ab7d84259324b94",
        "image_processing_paddleocr_vl.py": "06ca57dcf44b4f439c926e451d8dbb0969db49aaa678f01ff9eca64e29583995",
        "inference.yml": "b70e21114423f553b433e977f16c1360092ac9f609e04bc5c10fed3a8a791a6e",
        "model.safetensors": "d557c9d8997ae57ed3b1b33bdf347be878cc335687f32ca105341c16973f8958",
        "modeling_paddleocr_vl.py": "c5013dff57ca8b87dc1de64d0fd839a44313de09d230a4fb2d08289d2cad5111",
        "preprocessor_config.json": "111872ab1e8bb7fd040ac5087bfced7ab8f011f02139b088cba294964c3b1d0e",
        "processing_paddleocr_vl.py": "e29cb1e5f275f2bd3ce051bd5c9983a33894e693b2823a0e13d4c07c8c4f9e13",
        "processor_config.json": "1568858960a9760c54431dae693a6152e601ff55cdf6d2eab97a4a99958faea0",
        "special_tokens_map.json": "d3a125c03103deb2acaf7730791bdbbf196f620e5a2213b664511ff9b4b25bab",
        "tokenizer.json": "c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d",
        "tokenizer.model": "34ef7db83df785924fb83d7b887b6e822a031c56e15cff40aaf9b982988180df",
        "tokenizer_config.json": "1f979337347cc0cb72a6282d8a23ed183539aa81a87a906f022aee2bab83c7c5",
    },
)
DEFAULT_GGUF_RUNTIME_BUNDLE = PinnedSnapshotBundle(
    repo_id=DEFAULT_GGUF_MODEL_ID,
    revision="c8806b09a259ad6aaa7f401ed73dad2ff8df2c51",
    file_hashes={
        DEFAULT_GGUF_FILENAME: "299051d54faa065abc505cc39b8383ea338fd3020c775ea3e0ba514a7022328c",
        DEFAULT_GGUF_MMPROJ_FILENAME: "e7f1a72400fba517046f90d964e2fa0f4dac7781ee3b1bc5d2022f5f8cecbd87",
    },
)
DEFAULT_QWEN_CORRECTOR_BUNDLE = PinnedSnapshotBundle(
    repo_id=DEFAULT_QWEN_CORRECTOR_MODEL_ID,
    revision="bc014a17be43adabd7066b7a86075ff935c6a4e2",
    file_hashes={
        DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME: "1b0ac637dfa092bbba2793977db9485a40c4f8b42df5fe342f0076d61b66ae83",
        DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME: "abe81a7212be307a7723ab47a51a87e5c46d0622273ccb04a6a6feba18b21d63",
    },
)


def default_models_dir() -> Path:
    configured = os.environ.get("ISTOTS_MODELS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "istots" / "models").resolve()


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def managed_setup_target_marker_path(target: Path) -> Path:
    return target / MANAGED_SETUP_TARGET_SENTINEL


def is_default_pinned_hf_model(model_id: str) -> bool:
    return model_id == DEFAULT_HF_MODEL_BUNDLE.repo_id


def is_default_pinned_gguf_model(model_id: str) -> bool:
    return model_id == DEFAULT_GGUF_RUNTIME_BUNDLE.repo_id


def is_default_pinned_qwen_bundle(
    *,
    model_id: str,
    model_filename: str,
    mmproj_filename: str,
) -> bool:
    return (
        model_id == DEFAULT_QWEN_CORRECTOR_BUNDLE.repo_id
        and model_filename == DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME
        and mmproj_filename == DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME
    )


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


def _mark_setup_target_managed(
    target: Path,
    *,
    model_id: str,
    revision: str | None = None,
    verification: str = "unverified",
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    lines = [
        "managed_by=istots_setup",
        f"repo_id={model_id}",
        f"verification={verification}",
    ]
    if revision is not None:
        lines.append(f"revision={revision}")
    managed_setup_target_marker_path(target).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_download_to_target(
    *,
    repo_id: str,
    target: Path,
    force: bool = False,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for `istots setup`.") from exc

    target.parent.mkdir(parents=True, exist_ok=True)

    if force and target.exists():
        _ensure_safe_force_delete_target(target)
        shutil.rmtree(target)

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(target),
        "local_dir_use_symlinks": False,
        "local_files_only": False,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    snapshot_download(**kwargs)


def _verify_pinned_snapshot_files(
    *,
    target: Path,
    bundle: PinnedSnapshotBundle,
) -> dict[str, Path]:
    resolved_paths: dict[str, Path] = {}
    for relative_path, expected_sha256 in bundle.file_hashes.items():
        path = (target / relative_path).resolve()
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"pinned setup file missing after download: {path}")
        actual_sha256 = _sha256_path(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "pinned setup file failed hash check for "
                f"{relative_path}: expected {expected_sha256}, got {actual_sha256}"
            )
        resolved_paths[relative_path] = path
    return resolved_paths


def _download_pinned_snapshot_bundle(
    *,
    bundle: PinnedSnapshotBundle,
    models_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, dict[str, Path]]:
    revision = bundle.revision.strip()
    if not revision:
        raise RuntimeError(f"pinned setup bundle is missing a revision for {bundle.repo_id}")

    target = resolve_setup_download_path(model_id=bundle.repo_id, models_dir=models_dir)
    _snapshot_download_to_target(
        repo_id=bundle.repo_id,
        target=target,
        force=force,
        revision=revision,
        allow_patterns=list(bundle.file_hashes),
    )
    resolved_paths = _verify_pinned_snapshot_files(target=target, bundle=bundle)
    _mark_setup_target_managed(
        target,
        model_id=bundle.repo_id,
        revision=revision,
        verification="pinned",
    )
    return target.resolve(), resolved_paths


def resolve_local_model_path(model_id: str, models_dir: Path | None = None) -> Path:
    direct = Path(model_id).expanduser()
    if direct.exists():
        return direct.resolve()

    root = (models_dir or default_models_dir()).expanduser().resolve()
    return root / model_dir_name(model_id)


def ensure_local_model(model_id: str, models_dir: Path | None = None) -> Path:
    path = resolve_local_model_path(model_id=model_id, models_dir=models_dir)
    if not path.exists() or not (path / "config.json").exists():
        setup_command = "istots setup --with-hf-fallback"
        if not is_default_pinned_hf_model(model_id):
            setup_command = f"{setup_command} --model-id {model_id}"
        raise RuntimeError(
            "Model is not available locally. "
            f"Expected at: {path}. Run `uv sync --extra hf` and `{setup_command}` first."
        )
    return path


def download_model(model_id: str, models_dir: Path | None = None, force: bool = False) -> Path:
    if is_default_pinned_hf_model(model_id):
        target, _ = _download_pinned_snapshot_bundle(
            bundle=DEFAULT_HF_MODEL_BUNDLE,
            models_dir=models_dir,
            force=force,
        )
        return target

    target = resolve_setup_download_path(model_id=model_id, models_dir=models_dir)
    _snapshot_download_to_target(
        repo_id=model_id,
        target=target,
        force=force,
    )
    _mark_setup_target_managed(target, model_id=model_id, verification="unverified")
    return target.resolve()


def _download_snapshot_files(
    *,
    model_id: str,
    required_filenames: tuple[str, ...],
    models_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, tuple[Path, ...]]:
    target = resolve_setup_download_path(model_id=model_id, models_dir=models_dir)
    _snapshot_download_to_target(
        repo_id=model_id,
        target=target,
        force=force,
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
    _mark_setup_target_managed(target, model_id=model_id, verification="unverified")
    return target.resolve(), tuple(resolved_paths)


def download_gguf_runtime_assets(
    *,
    model_id: str = DEFAULT_GGUF_MODEL_ID,
    models_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    if is_default_pinned_gguf_model(model_id):
        target, resolved_paths = _download_pinned_snapshot_bundle(
            bundle=DEFAULT_GGUF_RUNTIME_BUNDLE,
            models_dir=models_dir,
            force=force,
        )
        return (
            target,
            resolved_paths[DEFAULT_GGUF_FILENAME],
            resolved_paths[DEFAULT_GGUF_MMPROJ_FILENAME],
        )

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
    if is_default_pinned_qwen_bundle(
        model_id=model_id,
        model_filename=model_filename,
        mmproj_filename=mmproj_filename,
    ):
        target, resolved_paths = _download_pinned_snapshot_bundle(
            bundle=DEFAULT_QWEN_CORRECTOR_BUNDLE,
            models_dir=models_dir,
            force=force,
        )
        return (
            target,
            resolved_paths[DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME],
            resolved_paths[DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME],
        )

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
    with_hf_fallback: bool = False,
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

    if not with_hf_fallback and not is_default_pinned_hf_model(hf_model_id):
        raise RuntimeError("Custom HF fallback model ids require `with_hf_fallback=True`.")

    hf_model_dir: Path | None = None
    if with_hf_fallback:
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
