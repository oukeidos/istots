from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from istots.derived_assets import resolve_derived_mmproj_output_path
from istots import model_store


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_model_dir_name_replaces_slash() -> None:
    assert model_store.model_dir_name("PaddlePaddle/PaddleOCR-VL-1.5") == "PaddlePaddle__PaddleOCR-VL-1.5"


def test_resolve_local_model_path_uses_cache_root(tmp_path: Path) -> None:
    path = model_store.resolve_local_model_path("org/model", models_dir=tmp_path)
    assert path == (tmp_path / "org__model").resolve()


def test_resolve_setup_download_path_uses_cache_root(tmp_path: Path) -> None:
    path = model_store.resolve_setup_download_path("org/model", models_dir=tmp_path)
    assert path == (tmp_path / "org__model").resolve()


def test_ensure_local_model_accepts_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "local_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    resolved = model_store.ensure_local_model(str(model_dir))
    assert resolved == model_dir.resolve()


def test_ensure_local_model_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Model is not available locally"):
        model_store.ensure_local_model("org/model", models_dir=tmp_path)


def test_download_default_gguf_runtime_assets_use_pinned_bundle(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    target = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    fake_bundle = model_store.PinnedSnapshotBundle(
        repo_id=model_store.DEFAULT_GGUF_MODEL_ID,
        revision="gguf-revision",
        file_hashes={
            model_store.DEFAULT_GGUF_FILENAME: _sha256_bytes(b"model"),
            model_store.DEFAULT_GGUF_MMPROJ_FILENAME: _sha256_bytes(b"mmproj"),
        },
    )

    def fake_download_pinned_bundle_files(*, bundle, target, cancel_callback=None) -> None:
        captured["bundle"] = bundle
        captured["target"] = target
        target.mkdir(parents=True, exist_ok=True)
        (target / model_store.DEFAULT_GGUF_FILENAME).write_bytes(b"model")
        (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).write_bytes(b"mmproj")

    monkeypatch.setattr(model_store, "_download_pinned_bundle_files", fake_download_pinned_bundle_files)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: pytest.fail("snapshot_download should not run")),
    )
    monkeypatch.setattr(model_store, "DEFAULT_GGUF_RUNTIME_BUNDLE", fake_bundle)

    gguf_dir, model_path, mmproj_path = model_store.download_gguf_runtime_assets(models_dir=tmp_path)

    assert gguf_dir == target.resolve()
    assert model_path == (target / model_store.DEFAULT_GGUF_FILENAME).resolve()
    assert mmproj_path == (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).resolve()
    assert captured["bundle"] == fake_bundle
    assert Path(captured["target"]).parent == tmp_path.resolve()
    marker = model_store.managed_setup_target_marker_path(target).read_text(encoding="utf-8")
    assert "verification=pinned\n" in marker
    assert "revision=gguf-revision\n" in marker


def test_download_custom_gguf_runtime_assets_remain_unverified(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    target = tmp_path / "org__model"
    target.mkdir(parents=True, exist_ok=True)
    (target / model_store.DEFAULT_GGUF_FILENAME).write_bytes(b"model")
    (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).write_bytes(b"mmproj")

    def fake_snapshot_download(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    gguf_dir, model_path, mmproj_path = model_store.download_gguf_runtime_assets(
        model_id="org/model",
        models_dir=tmp_path,
    )

    assert gguf_dir == target.resolve()
    assert model_path == (target / model_store.DEFAULT_GGUF_FILENAME).resolve()
    assert mmproj_path == (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).resolve()
    assert captured["repo_id"] == "org/model"
    assert "revision" not in captured
    assert captured["allow_patterns"] == [
        model_store.DEFAULT_GGUF_FILENAME,
        model_store.DEFAULT_GGUF_MMPROJ_FILENAME,
        "LICENSE*",
        "README*",
        "*.json",
    ]
    marker = model_store.managed_setup_target_marker_path(target).read_text(encoding="utf-8")
    assert "verification=unverified\n" in marker


def test_download_default_qwen_corrector_assets_use_pinned_bundle(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    target = tmp_path / "unsloth__Qwen3.5-35B-A3B-GGUF"
    fake_bundle = model_store.PinnedSnapshotBundle(
        repo_id=model_store.DEFAULT_QWEN_CORRECTOR_MODEL_ID,
        revision="qwen-revision",
        file_hashes={
            model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME: _sha256_bytes(b"model"),
            model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME: _sha256_bytes(b"mmproj"),
        },
    )

    def fake_download_pinned_bundle_files(*, bundle, target, cancel_callback=None) -> None:
        captured["bundle"] = bundle
        captured["target"] = target
        target.mkdir(parents=True, exist_ok=True)
        (target / model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME).write_bytes(b"model")
        (target / model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME).write_bytes(b"mmproj")

    monkeypatch.setattr(model_store, "_download_pinned_bundle_files", fake_download_pinned_bundle_files)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: pytest.fail("snapshot_download should not run")),
    )
    monkeypatch.setattr(model_store, "DEFAULT_QWEN_CORRECTOR_BUNDLE", fake_bundle)

    qwen_dir, model_path, mmproj_path = model_store.download_qwen_corrector_assets(models_dir=tmp_path)

    assert qwen_dir == target.resolve()
    assert model_path == (target / model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME).resolve()
    assert mmproj_path == (target / model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME).resolve()
    assert captured["bundle"] == fake_bundle
    assert Path(captured["target"]).parent == tmp_path.resolve()
    marker = model_store.managed_setup_target_marker_path(target).read_text(encoding="utf-8")
    assert "verification=pinned\n" in marker
    assert "revision=qwen-revision\n" in marker


def test_download_default_hf_model_uses_pinned_bundle(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    target = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5"
    fake_bundle = model_store.PinnedSnapshotBundle(
        repo_id=model_store.DEFAULT_MODEL_ID,
        revision="hf-revision",
        file_hashes={
            "config.json": _sha256_bytes(b"{}"),
            "tokenizer.json": _sha256_bytes(b"tokenizer"),
        },
    )

    def fake_download_pinned_bundle_files(*, bundle, target, cancel_callback=None) -> None:
        captured["bundle"] = bundle
        captured["target"] = target
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "tokenizer.json").write_text("tokenizer", encoding="utf-8")

    monkeypatch.setattr(model_store, "_download_pinned_bundle_files", fake_download_pinned_bundle_files)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: pytest.fail("snapshot_download should not run")),
    )
    monkeypatch.setattr(model_store, "DEFAULT_HF_MODEL_BUNDLE", fake_bundle)

    resolved = model_store.download_model(model_store.DEFAULT_MODEL_ID, models_dir=tmp_path)

    assert resolved == target.resolve()
    assert captured["bundle"] == fake_bundle
    assert Path(captured["target"]).parent == tmp_path.resolve()
    marker = model_store.managed_setup_target_marker_path(target).read_text(encoding="utf-8")
    assert "verification=pinned\n" in marker
    assert "revision=hf-revision\n" in marker


def test_pinned_bundle_download_url_quotes_repo_and_file_path() -> None:
    bundle = model_store.PinnedSnapshotBundle(
        repo_id="org name/model name",
        revision="abc123",
        file_hashes={"nested dir/file name.gguf": _sha256_bytes(b"payload")},
    )

    url = model_store._pinned_bundle_download_url(
        bundle=bundle,
        relative_path="nested dir/file name.gguf",
    )

    assert url == (
        "https://huggingface.co/org%20name/model%20name/resolve/abc123/"
        "nested%20dir/file%20name.gguf"
    )


def test_download_custom_hf_model_marks_target_as_unverified(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs) -> None:
        captured.update(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    target = model_store.download_model("org/model", models_dir=tmp_path)

    assert target == (tmp_path / "org__model").resolve()
    assert captured["repo_id"] == "org/model"
    assert "revision" not in captured
    marker = model_store.managed_setup_target_marker_path(target).read_text(encoding="utf-8")
    assert "verification=unverified\n" in marker


def test_download_model_disables_xet_during_snapshot_download(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    constants_module = SimpleNamespace(HF_HUB_DISABLE_XET=False)

    def fake_snapshot_download(**kwargs) -> None:
        captured.update(kwargs)
        captured["env_disable_xet"] = os.environ.get("HF_HUB_DISABLE_XET")
        captured["constants_disable_xet"] = constants_module.HF_HUB_DISABLE_XET
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants_module)

    target = model_store.download_model("org/model", models_dir=tmp_path)

    assert target == (tmp_path / "org__model").resolve()
    assert captured["env_disable_xet"] == "1"
    assert captured["constants_disable_xet"] is True
    assert "HF_HUB_DISABLE_XET" not in os.environ
    assert constants_module.HF_HUB_DISABLE_XET is False


def test_download_model_restores_existing_xet_environment_override(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    constants_module = SimpleNamespace(HF_HUB_DISABLE_XET=False)

    def fake_snapshot_download(**kwargs) -> None:
        captured["env_disable_xet"] = os.environ.get("HF_HUB_DISABLE_XET")
        captured["constants_disable_xet"] = constants_module.HF_HUB_DISABLE_XET
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants_module)

    model_store.download_model("org/model", models_dir=tmp_path)

    assert captured["env_disable_xet"] == "1"
    assert captured["constants_disable_xet"] is True
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"
    assert constants_module.HF_HUB_DISABLE_XET is False


def test_download_pinned_bundle_rejects_missing_revision(monkeypatch, tmp_path: Path) -> None:
    fake_bundle = model_store.PinnedSnapshotBundle(
        repo_id=model_store.DEFAULT_GGUF_MODEL_ID,
        revision="",
        file_hashes={
            model_store.DEFAULT_GGUF_FILENAME: _sha256_bytes(b"model"),
            model_store.DEFAULT_GGUF_MMPROJ_FILENAME: _sha256_bytes(b"mmproj"),
        },
    )

    monkeypatch.setattr(model_store, "DEFAULT_GGUF_RUNTIME_BUNDLE", fake_bundle)

    with pytest.raises(RuntimeError, match="missing a revision"):
        model_store.download_gguf_runtime_assets(models_dir=tmp_path)


def test_download_pinned_bundle_rejects_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    fake_bundle = model_store.PinnedSnapshotBundle(
        repo_id=model_store.DEFAULT_GGUF_MODEL_ID,
        revision="gguf-revision",
        file_hashes={
            model_store.DEFAULT_GGUF_FILENAME: _sha256_bytes(b"expected-model"),
            model_store.DEFAULT_GGUF_MMPROJ_FILENAME: _sha256_bytes(b"expected-mmproj"),
        },
    )

    def fake_download_pinned_bundle_files(*, bundle, target, cancel_callback=None) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / model_store.DEFAULT_GGUF_FILENAME).write_bytes(b"wrong-model")
        (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).write_bytes(b"expected-mmproj")

    monkeypatch.setattr(model_store, "_download_pinned_bundle_files", fake_download_pinned_bundle_files)
    monkeypatch.setattr(model_store, "DEFAULT_GGUF_RUNTIME_BUNDLE", fake_bundle)

    with pytest.raises(RuntimeError, match="failed hash check"):
        model_store.download_gguf_runtime_assets(models_dir=tmp_path)


def test_download_model_rejects_existing_local_path_model_id(tmp_path: Path) -> None:
    local_dir = tmp_path / "existing-model-dir"
    local_dir.mkdir()

    with pytest.raises(RuntimeError, match="must be a Hugging Face repo ID"):
        model_store.download_model(str(local_dir), force=True)


def test_download_model_force_rejects_unmanaged_existing_target(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "org__model"
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    called = False

    def fake_snapshot_download(**kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    with pytest.raises(RuntimeError, match="not marked as istots-managed"):
        model_store.download_model("org/model", models_dir=tmp_path, force=True)

    assert called is False
    assert target.exists()


def test_download_model_force_allows_managed_existing_target(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "org__model"
    target.mkdir(parents=True, exist_ok=True)
    marker = model_store.managed_setup_target_marker_path(target)
    marker.write_text("managed_by=istots_setup\nrepo_id=org/model\n", encoding="utf-8")
    deleted: list[Path] = []
    captured: dict[str, object] = {}
    original_rmtree = shutil.rmtree

    def fake_rmtree(path: str | Path) -> None:
        deleted.append(Path(path).resolve())
        original_rmtree(path)

    def fake_snapshot_download(**kwargs) -> None:
        captured.update(kwargs)
        restored_target = Path(kwargs["local_dir"])
        restored_target.mkdir(parents=True, exist_ok=True)
        (restored_target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(model_store.shutil, "rmtree", fake_rmtree)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    result = model_store.download_model("org/model", models_dir=tmp_path, force=True)

    assert result == target.resolve()
    assert deleted == [target.resolve()]
    assert captured["local_dir"] == str(target.resolve())
    assert model_store.managed_setup_target_marker_path(target).exists()


def test_download_gguf_runtime_assets_force_rejects_unmanaged_existing_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    target.mkdir(parents=True, exist_ok=True)
    (target / "unexpected.txt").write_text("user file", encoding="utf-8")
    called = False

    def fake_download_pinned_bundle_files(*, bundle, target, cancel_callback=None) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(model_store, "_download_pinned_bundle_files", fake_download_pinned_bundle_files)

    with pytest.raises(RuntimeError, match="not clearly app-managed"):
        model_store.download_gguf_runtime_assets(models_dir=tmp_path, force=True)

    assert called is False
    assert target.exists()


def test_download_pinned_bundle_replaces_incomplete_expected_target(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    target.mkdir(parents=True, exist_ok=True)
    (target / model_store.DEFAULT_GGUF_FILENAME).write_bytes(b"partial-model")
    fake_bundle = model_store.PinnedSnapshotBundle(
        repo_id=model_store.DEFAULT_GGUF_MODEL_ID,
        revision="gguf-revision",
        file_hashes={
            model_store.DEFAULT_GGUF_FILENAME: _sha256_bytes(b"model"),
            model_store.DEFAULT_GGUF_MMPROJ_FILENAME: _sha256_bytes(b"mmproj"),
        },
    )

    def fake_download_pinned_bundle_files(*, bundle, target, cancel_callback=None) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / model_store.DEFAULT_GGUF_FILENAME).write_bytes(b"model")
        (target / model_store.DEFAULT_GGUF_MMPROJ_FILENAME).write_bytes(b"mmproj")

    monkeypatch.setattr(model_store, "_download_pinned_bundle_files", fake_download_pinned_bundle_files)
    monkeypatch.setattr(model_store, "DEFAULT_GGUF_RUNTIME_BUNDLE", fake_bundle)

    resolved_dir, model_path, mmproj_path = model_store.download_gguf_runtime_assets(models_dir=tmp_path)

    assert resolved_dir == target.resolve()
    assert model_path.read_bytes() == b"model"
    assert mmproj_path.read_bytes() == b"mmproj"
    marker = model_store.managed_setup_target_marker_path(target).read_text(encoding="utf-8")
    assert "verification=pinned\n" in marker


def test_ensure_local_qwen_corrector_assets_resolves_default_download(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "unsloth__Qwen3.5-35B-A3B-GGUF"
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME
    mmproj_path = target / model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME
    model_path.write_bytes(b"model")
    mmproj_path.write_bytes(b"mmproj")

    resolved_model_path, resolved_mmproj_path = model_store.ensure_local_qwen_corrector_assets(
        models_dir=tmp_path,
    )

    assert resolved_model_path == model_path.resolve()
    assert resolved_mmproj_path == mmproj_path.resolve()


def test_ensure_local_qwen_corrector_assets_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Run `istots setup --with-qwen-corrector`"):
        model_store.ensure_local_qwen_corrector_assets(models_dir=tmp_path)


def test_setup_default_runtime_assets_downloads_and_materializes_without_hf_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    gguf_dir = tmp_path / "gguf_model"
    gguf_model_path = gguf_dir / model_store.DEFAULT_GGUF_FILENAME
    gguf_mmproj_path = gguf_dir / model_store.DEFAULT_GGUF_MMPROJ_FILENAME
    derived_path = resolve_derived_mmproj_output_path(
        base_mmproj=gguf_mmproj_path,
        models_dir=tmp_path,
        min_pixels=32768,
    )

    monkeypatch.setattr(
        model_store,
        "download_model",
        lambda model_id, models_dir=None, force=False, cancel_callback=None: pytest.fail(
            "download_model should not run without `with_hf_fallback=True`"
        ),
    )
    monkeypatch.setattr(
        model_store,
        "download_gguf_runtime_assets",
        lambda model_id=model_store.DEFAULT_GGUF_MODEL_ID, models_dir=None, force=False, cancel_callback=None: (
            gguf_dir,
            gguf_model_path,
            gguf_mmproj_path,
        ),
    )

    calls: dict[str, object] = {}

    def fake_materialize_mmproj(
        *,
        base_mmproj,
        min_pixels,
        support_dir=None,
        gguf_py_base_url=None,
        gguf_source_mode="auto",
        force=False,
        output_path=None,
    ):
        calls["base_mmproj"] = base_mmproj
        calls["min_pixels"] = min_pixels
        calls["gguf_source_mode"] = gguf_source_mode
        calls["force"] = force
        calls["output_path"] = output_path
        return derived_path

    monkeypatch.setattr(
        "istots.llama_mmproj.materialize_mmproj",
        fake_materialize_mmproj,
    )

    artifacts = model_store.setup_default_runtime_assets(
        models_dir=tmp_path,
        force=True,
        gguf_source_mode="auto",
    )

    assert artifacts.hf_model_dir is None
    assert artifacts.gguf_model_dir == gguf_dir
    assert artifacts.gguf_model_path == gguf_model_path
    assert artifacts.gguf_mmproj_path == gguf_mmproj_path
    assert artifacts.gguf_mmproj_minpix32768_path == derived_path
    assert calls == {
        "base_mmproj": gguf_mmproj_path,
        "min_pixels": 32768,
        "gguf_source_mode": "auto",
        "force": True,
        "output_path": derived_path,
    }


def test_setup_default_runtime_assets_optionally_downloads_hf_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    hf_dir = tmp_path / "hf_model"
    gguf_dir = tmp_path / "gguf_model"
    gguf_model_path = gguf_dir / model_store.DEFAULT_GGUF_FILENAME
    gguf_mmproj_path = gguf_dir / model_store.DEFAULT_GGUF_MMPROJ_FILENAME
    derived_path = resolve_derived_mmproj_output_path(
        base_mmproj=gguf_mmproj_path,
        models_dir=tmp_path,
        min_pixels=32768,
    )
    calls: dict[str, object] = {}

    def fake_download_model(model_id, models_dir=None, force=False, cancel_callback=None):
        calls["download_model"] = {
            "model_id": model_id,
            "models_dir": models_dir,
            "force": force,
            "cancel_callback": cancel_callback,
        }
        return hf_dir

    monkeypatch.setattr(model_store, "download_model", fake_download_model)
    monkeypatch.setattr(
        model_store,
        "download_gguf_runtime_assets",
        lambda model_id=model_store.DEFAULT_GGUF_MODEL_ID, models_dir=None, force=False, cancel_callback=None: (
            gguf_dir,
            gguf_model_path,
            gguf_mmproj_path,
        ),
    )
    monkeypatch.setattr(
        "istots.llama_mmproj.materialize_mmproj",
        lambda **kwargs: derived_path,
    )

    artifacts = model_store.setup_default_runtime_assets(
        models_dir=tmp_path,
        force=True,
        with_hf_fallback=True,
    )

    assert artifacts.hf_model_dir == hf_dir
    assert calls["download_model"] == {
        "model_id": model_store.DEFAULT_MODEL_ID,
        "models_dir": tmp_path,
        "force": True,
        "cancel_callback": None,
    }


def test_setup_default_runtime_assets_optionally_downloads_qwen_corrector(monkeypatch, tmp_path: Path) -> None:
    gguf_dir = tmp_path / "gguf_model"
    gguf_model_path = gguf_dir / model_store.DEFAULT_GGUF_FILENAME
    gguf_mmproj_path = gguf_dir / model_store.DEFAULT_GGUF_MMPROJ_FILENAME
    derived_path = resolve_derived_mmproj_output_path(
        base_mmproj=gguf_mmproj_path,
        models_dir=tmp_path,
        min_pixels=32768,
    )
    qwen_dir = tmp_path / "qwen_model"
    qwen_model_path = qwen_dir / model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME
    qwen_mmproj_path = qwen_dir / model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME

    monkeypatch.setattr(
        model_store,
        "download_model",
        lambda model_id, models_dir=None, force=False, cancel_callback=None: pytest.fail(
            "download_model should not run without `with_hf_fallback=True`"
        ),
    )
    monkeypatch.setattr(
        model_store,
        "download_gguf_runtime_assets",
        lambda model_id=model_store.DEFAULT_GGUF_MODEL_ID, models_dir=None, force=False, cancel_callback=None: (
            gguf_dir,
            gguf_model_path,
            gguf_mmproj_path,
        ),
    )
    qwen_calls: dict[str, object] = {}

    def fake_download_qwen_corrector_assets(
        *,
        model_id=model_store.DEFAULT_QWEN_CORRECTOR_MODEL_ID,
            model_filename=model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME,
            mmproj_filename=model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME,
            models_dir=None,
            force=False,
            cancel_callback=None,
        ):
        qwen_calls.update(
            {
                "model_id": model_id,
                "model_filename": model_filename,
                "mmproj_filename": mmproj_filename,
                "models_dir": models_dir,
                "force": force,
                "cancel_callback": cancel_callback,
            }
        )
        return qwen_dir, qwen_model_path, qwen_mmproj_path

    monkeypatch.setattr(model_store, "download_qwen_corrector_assets", fake_download_qwen_corrector_assets)
    monkeypatch.setattr(
        "istots.llama_mmproj.materialize_mmproj",
        lambda **kwargs: derived_path,
    )

    artifacts = model_store.setup_default_runtime_assets(
        models_dir=tmp_path,
        force=True,
        with_qwen_corrector=True,
    )

    assert artifacts.qwen_corrector_dir == qwen_dir
    assert artifacts.qwen_corrector_model_path == qwen_model_path
    assert artifacts.qwen_corrector_mmproj_path == qwen_mmproj_path
    assert qwen_calls == {
        "model_id": model_store.DEFAULT_QWEN_CORRECTOR_MODEL_ID,
        "model_filename": model_store.DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME,
        "mmproj_filename": model_store.DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME,
        "models_dir": tmp_path,
        "force": True,
        "cancel_callback": None,
    }


def test_setup_default_runtime_assets_rejects_custom_hf_model_without_opt_in() -> None:
    with pytest.raises(RuntimeError, match="require `with_hf_fallback=True`"):
        model_store.setup_default_runtime_assets(hf_model_id="org/model")
