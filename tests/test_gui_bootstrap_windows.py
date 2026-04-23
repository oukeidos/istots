from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from istots.gui import bootstrap_windows


def test_auto_runtime_variant_candidates_prioritize_cuda_then_vulkan_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        bootstrap_windows,
        "_can_load_system_library",
        lambda name: name in {"nvcuda.dll", "vulkan-1.dll"},
    )

    assert bootstrap_windows.auto_runtime_variant_candidates() == (
        "x64/cuda12",
        "x64/vulkan",
        "x64/cpu",
    )


def test_resolve_runtime_variant_skips_missing_auto_assets(monkeypatch) -> None:
    catalog = bootstrap_windows.LlamaCppReleaseCatalog(
        tag_name="b8855",
        assets=(
            bootstrap_windows.LlamaCppReleaseAsset(
                name="llama-b8855-bin-win-vulkan-x64.zip",
                download_url="https://example.invalid/vulkan.zip",
                size_bytes=1,
                sha256_digest="0" * 64,
            ),
            bootstrap_windows.LlamaCppReleaseAsset(
                name="llama-b8855-bin-win-cpu-x64.zip",
                download_url="https://example.invalid/cpu.zip",
                size_bytes=1,
                sha256_digest="1" * 64,
            ),
        ),
    )
    monkeypatch.setattr(
        bootstrap_windows,
        "auto_runtime_variant_candidates",
        lambda: ("x64/cuda12", "x64/vulkan", "x64/cpu"),
    )

    assert bootstrap_windows.resolve_runtime_variant(catalog, requested_variant="auto") == "x64/vulkan"


def test_select_allowlisted_runtime_candidates_rejects_unallowlisted_manual_variant() -> None:
    with pytest.raises(RuntimeError, match="unsupported managed llama.cpp runtime variant"):
        bootstrap_windows.select_allowlisted_runtime_candidates(requested_variant="arm64/cpu")


def test_select_allowlisted_runtime_candidates_prioritize_never_tried_tags_then_lower_attempt_counts() -> None:
    attempt_history = {
        ("x64/cpu", "b8887"): bootstrap_windows.ManagedRuntimeAttemptRecord(
            release_tag="b8887",
            variant_id="x64/cpu",
            attempt_count=2,
        ),
        ("x64/cpu", "b8886"): bootstrap_windows.ManagedRuntimeAttemptRecord(
            release_tag="b8886",
            variant_id="x64/cpu",
            attempt_count=1,
        ),
    }

    candidates = bootstrap_windows.select_allowlisted_runtime_candidates(
        requested_variant="x64/cpu",
        attempt_history=attempt_history,
    )

    assert candidates[:3] == (
        bootstrap_windows.ManagedRuntimeCandidate("b8885", "x64/cpu"),
        bootstrap_windows.ManagedRuntimeCandidate("b8833", "x64/cpu"),
        bootstrap_windows.ManagedRuntimeCandidate("b8832", "x64/cpu"),
    )


def test_select_allowlisted_runtime_candidates_keep_cpu_slot_for_auto(monkeypatch) -> None:
    monkeypatch.setattr(
        bootstrap_windows,
        "auto_runtime_variant_candidates",
        lambda: ("x64/cuda12", "x64/vulkan", "x64/cpu"),
    )

    candidates = bootstrap_windows.select_allowlisted_runtime_candidates(requested_variant="auto")

    assert candidates == (
        bootstrap_windows.ManagedRuntimeCandidate("b8892", "x64/cuda12"),
        bootstrap_windows.ManagedRuntimeCandidate("b8892", "x64/vulkan"),
        bootstrap_windows.ManagedRuntimeCandidate("b8885", "x64/cuda12"),
        bootstrap_windows.ManagedRuntimeCandidate("b8887", "x64/cpu"),
    )


def test_resolve_gui_runtime_binding_prefers_managed_state(monkeypatch, tmp_path: Path) -> None:
    state = bootstrap_windows.ManagedLlamaCppRuntimeState(
        release_tag="b8855",
        variant_id="x64/cpu",
        install_dir=tmp_path / "runtime",
        binary_path=tmp_path / "runtime" / "llama-server.exe",
    )
    state.install_dir.mkdir(parents=True, exist_ok=True)
    state.binary_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_windows, "load_managed_runtime_state", lambda: state)
    monkeypatch.setattr(
        bootstrap_windows,
        "_detect_external_llama_server_path",
        lambda explicit: Path("/tmp/external/llama-server.exe"),
    )

    binding = bootstrap_windows.resolve_gui_runtime_binding()

    assert binding.source == bootstrap_windows.MANAGED_RUNTIME_SOURCE
    assert binding.binary_path == state.binary_path
    assert binding.release_tag == "b8855"
    assert binding.variant_id == "x64/cpu"


def test_resolve_gui_runtime_binding_falls_back_to_external_when_managed_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = bootstrap_windows.ManagedLlamaCppRuntimeState(
        release_tag="b8855",
        variant_id="x64/cpu",
        install_dir=tmp_path / "runtime",
        binary_path=tmp_path / "runtime" / "missing.exe",
        preferred_source=bootstrap_windows.EXTERNAL_RUNTIME_SOURCE,
    )
    external_binary = tmp_path / "external" / "llama-server.exe"
    external_binary.parent.mkdir(parents=True, exist_ok=True)
    external_binary.write_text("", encoding="utf-8")

    monkeypatch.setattr(bootstrap_windows, "load_managed_runtime_state", lambda: state)
    monkeypatch.setattr(
        bootstrap_windows,
        "_detect_external_llama_server_path",
        lambda explicit: external_binary,
    )

    binding = bootstrap_windows.resolve_gui_runtime_binding()

    assert binding.source == bootstrap_windows.EXTERNAL_RUNTIME_SOURCE
    assert binding.binary_path == external_binary.resolve()


def test_validate_llama_server_binary_accepts_zero_exit_probe(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")
    seen: list[tuple[list[str], dict[str, object]]] = []

    @contextmanager
    def _fake_subprocess_runtime():
        yield {"PATH": "clean"}

    def _fake_popen(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(args=command)

    monkeypatch.setattr(bootstrap_windows, "missing_managed_runtime_prerequisites", lambda: ())
    monkeypatch.setattr(bootstrap_windows, "sanitized_external_subprocess_runtime", _fake_subprocess_runtime)
    monkeypatch.setattr(bootstrap_windows.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        bootstrap_windows,
        "_wait_for_validation_process",
        lambda process, *, timeout, cancel_event: SimpleNamespace(returncode=0, stdout="version", stderr=""),
    )

    bootstrap_windows.validate_llama_server_binary(binary)

    assert seen[0][0] == [str(binary.resolve()), "--version"]
    assert seen[0][1]["creationflags"] & getattr(bootstrap_windows.subprocess, "CREATE_NO_WINDOW", 0)
    assert seen[0][1]["env"] == {"PATH": "clean"}


def test_validate_llama_server_binary_reports_failed_probes(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")
    seen: list[tuple[list[str], dict[str, object]]] = []

    @contextmanager
    def _fake_subprocess_runtime():
        yield {"PATH": "clean"}

    def _fake_popen(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(args=command)

    monkeypatch.setattr(bootstrap_windows, "missing_managed_runtime_prerequisites", lambda: ())
    monkeypatch.setattr(bootstrap_windows, "sanitized_external_subprocess_runtime", _fake_subprocess_runtime)
    monkeypatch.setattr(bootstrap_windows.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        bootstrap_windows,
        "_wait_for_validation_process",
        lambda process, *, timeout, cancel_event: SimpleNamespace(returncode=3221225477, stdout="", stderr=""),
    )

    try:
        bootstrap_windows.validate_llama_server_binary(binary)
    except RuntimeError as exc:
        message = str(exc)
        assert "managed llama.cpp runtime failed startup validation." in message
        assert "Probe --version failed with exit=3221225477." in message
    else:
        raise AssertionError("expected startup validation failure")

    assert [command for command, _ in seen] == [[str(binary.resolve()), "--version"]]
    assert all(
        kwargs["creationflags"] & getattr(bootstrap_windows.subprocess, "CREATE_NO_WINDOW", 0)
        for _, kwargs in seen
    )
    assert all(kwargs["env"] == {"PATH": "clean"} for _, kwargs in seen)


def test_validate_llama_server_binary_reports_missing_prerequisite(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")
    prerequisite = SimpleNamespace(
        label="Microsoft Visual C++ Redistributable (x64)",
        detail="not detected",
        remediation="install it",
        installer_url="https://example.invalid/vc_redist.x64.exe",
        installed_version=None,
    )

    monkeypatch.setattr(bootstrap_windows, "missing_managed_runtime_prerequisites", lambda: (prerequisite,))
    monkeypatch.setattr(
        bootstrap_windows,
        "format_missing_managed_runtime_prerequisites",
        lambda items: "Managed runtime prerequisite check failed.",
    )

    with pytest.raises(RuntimeError, match="Managed runtime prerequisite check failed."):
        bootstrap_windows.validate_llama_server_binary(binary)


def test_install_managed_runtime_does_not_reprobe_same_candidate_after_failed_existing_variant_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = bootstrap_windows.GuiManagedPaths(
        root=tmp_path / "managed",
        models_dir=tmp_path / "managed" / "models",
        runtime_dir=tmp_path / "managed" / "runtime" / "llama.cpp",
        derived_mmproj_dir=tmp_path / "managed" / "derived" / "mmproj",
        state_dir=tmp_path / "managed" / "state",
        runtime_state_path=tmp_path / "managed" / "state" / bootstrap_windows.GUI_RUNTIME_STATE_FILENAME,
    )
    variant_dir = paths.runtime_dir / "b8855" / "x64-vulkan"
    binary = variant_dir / "llama-server.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("", encoding="utf-8")
    catalog = bootstrap_windows.LlamaCppReleaseCatalog(
        tag_name="b8855",
        assets=(
            bootstrap_windows.LlamaCppReleaseAsset(
                name="llama-b8855-bin-win-vulkan-x64.zip",
                download_url="https://example.invalid/vulkan.zip",
                size_bytes=1,
                sha256_digest="0" * 64,
            ),
        ),
    )
    validations: list[Path] = []
    removals: list[Path] = []
    attempts: list[tuple[str, str, str]] = []

    monkeypatch.setattr(bootstrap_windows, "gui_managed_paths", lambda: paths)
    monkeypatch.setattr(bootstrap_windows, "ensure_managed_runtime_prerequisites", lambda **kwargs: ())
    monkeypatch.setattr(bootstrap_windows, "load_managed_runtime_state", lambda: None)
    monkeypatch.setattr(bootstrap_windows, "load_managed_runtime_attempt_history", lambda: {})
    monkeypatch.setattr(
        bootstrap_windows,
        "select_allowlisted_runtime_candidates",
        lambda **kwargs: (bootstrap_windows.ManagedRuntimeCandidate("b8855", "x64/vulkan"),),
    )
    monkeypatch.setattr(bootstrap_windows, "fetch_llama_cpp_release_by_tag", lambda *args, **kwargs: catalog)

    monkeypatch.setattr(bootstrap_windows, "_locate_llama_server_binary", lambda _install_dir: binary)
    monkeypatch.setattr(
        bootstrap_windows,
        "validate_llama_server_binary",
        lambda path, cancel_event=None: validations.append(path) or (_ for _ in ()).throw(RuntimeError("broken runtime")),
    )
    monkeypatch.setattr(
        bootstrap_windows,
        "_safe_rmtree",
        lambda path, *, within: removals.append(path),
    )
    monkeypatch.setattr(
        bootstrap_windows,
        "record_managed_runtime_attempt",
        lambda **kwargs: attempts.append((kwargs["release_tag"], kwargs["variant_id"], kwargs["outcome"])),
    )

    with pytest.raises(RuntimeError, match="failed after trying approved candidates"):
        bootstrap_windows.install_managed_llama_cpp_runtime(
            requested_variant="x64/vulkan",
            fetch_bytes=lambda url: b"",
        )

    assert validations == [binary]
    assert variant_dir in removals
    assert attempts == [("b8855", "x64/vulkan", "probe_failed")]


def test_install_managed_runtime_does_not_reuse_existing_state_for_different_requested_variant(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = bootstrap_windows.GuiManagedPaths(
        root=tmp_path / "managed",
        models_dir=tmp_path / "managed" / "models",
        runtime_dir=tmp_path / "managed" / "runtime" / "llama.cpp",
        derived_mmproj_dir=tmp_path / "managed" / "derived" / "mmproj",
        state_dir=tmp_path / "managed" / "state",
        runtime_state_path=tmp_path / "managed" / "state" / bootstrap_windows.GUI_RUNTIME_STATE_FILENAME,
    )
    existing_binary = paths.runtime_dir / "b8855" / "x64-vulkan" / "llama-server.exe"
    existing_binary.parent.mkdir(parents=True, exist_ok=True)
    existing_binary.write_text("", encoding="utf-8")
    existing_state = bootstrap_windows.ManagedLlamaCppRuntimeState(
        release_tag="b8855",
        variant_id="x64/vulkan",
        install_dir=existing_binary.parent,
        binary_path=existing_binary,
    )
    cpu_binary = paths.runtime_dir / "b8855" / "x64-cpu" / "llama-server.exe"
    writes: list[bootstrap_windows.ManagedLlamaCppRuntimeState] = []
    validations: list[Path] = []
    downloads: list[str] = []
    extractions: list[str] = []
    catalog = bootstrap_windows.LlamaCppReleaseCatalog(
        tag_name="b8855",
        assets=(
            bootstrap_windows.LlamaCppReleaseAsset(
                name="llama-b8855-bin-win-cpu-x64.zip",
                download_url="https://example.invalid/cpu.zip",
                size_bytes=1,
                sha256_digest="3" * 64,
            ),
        ),
    )

    monkeypatch.setattr(bootstrap_windows, "gui_managed_paths", lambda: paths)
    monkeypatch.setattr(bootstrap_windows, "ensure_managed_runtime_prerequisites", lambda **kwargs: ())
    monkeypatch.setattr(bootstrap_windows, "load_managed_runtime_state", lambda: existing_state)
    monkeypatch.setattr(bootstrap_windows, "load_managed_runtime_attempt_history", lambda: {})
    monkeypatch.setattr(bootstrap_windows, "write_managed_runtime_state", writes.append)
    monkeypatch.setattr(bootstrap_windows, "record_managed_runtime_attempt", lambda **kwargs: None)
    monkeypatch.setattr(
        bootstrap_windows,
        "select_allowlisted_runtime_candidates",
        lambda **kwargs: (bootstrap_windows.ManagedRuntimeCandidate("b8855", "x64/cpu"),),
    )
    monkeypatch.setattr(bootstrap_windows, "fetch_llama_cpp_release_by_tag", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(
        bootstrap_windows,
        "validate_llama_server_binary",
        lambda path, cancel_event=None: validations.append(path),
    )
    monkeypatch.setattr(
        bootstrap_windows,
        "_download_release_assets",
        lambda **kwargs: downloads.append("downloaded") or (kwargs["stage_dir"] / "cpu.zip",),
    )
    monkeypatch.setattr(
        bootstrap_windows,
        "_extract_release_archives",
        lambda **kwargs: extractions.append("extracted") or cpu_binary.parent.mkdir(parents=True, exist_ok=True),
    )
    locate_results = [None, cpu_binary]
    monkeypatch.setattr(
        bootstrap_windows,
        "_locate_llama_server_binary",
        lambda install_dir: (
            locate_results.pop(0)
            if install_dir == cpu_binary.parent and locate_results
            else (cpu_binary if install_dir == cpu_binary.parent else None)
        ),
    )

    state = bootstrap_windows.install_managed_llama_cpp_runtime(requested_variant="x64/cpu")

    assert downloads == ["downloaded"]
    assert extractions == ["extracted"]
    assert validations == [cpu_binary]
    assert state.variant_id == "x64/cpu"
    assert writes and writes[-1].variant_id == "x64/cpu"


def test_record_managed_runtime_validation_updates_state(monkeypatch, tmp_path: Path) -> None:
    state = bootstrap_windows.ManagedLlamaCppRuntimeState(
        release_tag="b8858",
        variant_id="x64/vulkan",
        install_dir=tmp_path / "runtime",
        binary_path=tmp_path / "runtime" / "llama-server.exe",
    )
    writes: list[bootstrap_windows.ManagedLlamaCppRuntimeState] = []

    monkeypatch.setattr(bootstrap_windows, "load_managed_runtime_state", lambda: state)
    monkeypatch.setattr(bootstrap_windows, "write_managed_runtime_state", writes.append)

    bootstrap_windows.record_managed_runtime_validation(
        ok=False,
        detail="runtime test failed",
        binary_path=state.binary_path,
    )

    assert writes
    assert writes[0].last_validation_ok is False
    assert writes[0].last_validation_detail == "runtime test failed"


def test_record_managed_runtime_attempt_persists_history(monkeypatch, tmp_path: Path) -> None:
    paths = bootstrap_windows.GuiManagedPaths(
        root=tmp_path / "managed",
        models_dir=tmp_path / "managed" / "models",
        runtime_dir=tmp_path / "managed" / "runtime" / "llama.cpp",
        derived_mmproj_dir=tmp_path / "managed" / "derived" / "mmproj",
        state_dir=tmp_path / "managed" / "state",
        runtime_state_path=tmp_path / "managed" / "state" / bootstrap_windows.GUI_RUNTIME_STATE_FILENAME,
    )
    monkeypatch.setattr(bootstrap_windows, "gui_managed_paths", lambda: paths)

    bootstrap_windows.record_managed_runtime_attempt(
        release_tag="b8887",
        variant_id="x64/cpu",
        outcome="probe_failed",
        detail="blocked",
    )
    bootstrap_windows.record_managed_runtime_attempt(
        release_tag="b8887",
        variant_id="x64/cpu",
        outcome="installed_ok",
        detail="validated",
    )

    history = bootstrap_windows.load_managed_runtime_attempt_history()

    assert history[("x64/cpu", "b8887")].attempt_count == 2
    assert history[("x64/cpu", "b8887")].last_outcome == "installed_ok"
    assert history[("x64/cpu", "b8887")].last_detail == "validated"


def test_fetch_latest_llama_cpp_release_parses_asset_sha256_digest() -> None:
    catalog = bootstrap_windows.fetch_latest_llama_cpp_release(
        fetch_bytes=lambda url: (
            b'{"tag_name":"b9001","assets":[{"name":"llama-b9001-bin-win-cpu-x64.zip",'
            b'"browser_download_url":"https://example.invalid/cpu.zip","size":7,'
            b'"digest":"sha256:ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789"}]}'
        ),
    )

    assert catalog.tag_name == "b9001"
    assert catalog.assets[0].sha256_digest == "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"


def test_select_release_assets_requires_sha256_digest_for_selected_asset() -> None:
    catalog = bootstrap_windows.LlamaCppReleaseCatalog(
        tag_name="b9002",
        assets=(
            bootstrap_windows.LlamaCppReleaseAsset(
                name="llama-b9002-bin-win-cpu-x64.zip",
                download_url="https://example.invalid/cpu.zip",
                size_bytes=1,
                sha256_digest=None,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="does not expose a verified SHA-256 digest"):
        bootstrap_windows.select_release_assets(catalog, "x64/cpu")


def test_download_release_assets_rejects_sha256_mismatch(tmp_path: Path, monkeypatch) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    asset = bootstrap_windows.LlamaCppReleaseAsset(
        name="llama-b9003-bin-win-cpu-x64.zip",
        download_url="https://example.invalid/cpu.zip",
        size_bytes=7,
        sha256_digest="0" * 64,
    )

    class _Response:
        headers = {"Content-Length": "7"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_done"):
                return b""
            self._done = True
            return b"payload"

    monkeypatch.setattr(
        bootstrap_windows.urllib.request,
        "urlopen",
        lambda request, timeout=300: _Response(),
    )

    with pytest.raises(RuntimeError, match="failed SHA-256 verification"):
        bootstrap_windows._download_release_assets(
            assets=(asset,),
            stage_dir=stage_dir,
            progress_callback=None,
        )


def test_download_release_assets_accepts_matching_sha256(tmp_path: Path, monkeypatch) -> None:
    payload = b"payload"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    asset = bootstrap_windows.LlamaCppReleaseAsset(
        name="llama-b9004-bin-win-cpu-x64.zip",
        download_url="https://example.invalid/cpu.zip",
        size_bytes=len(payload),
        sha256_digest=hashlib.sha256(payload).hexdigest(),
    )

    class _Response:
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_done"):
                return b""
            self._done = True
            return payload

    monkeypatch.setattr(
        bootstrap_windows.urllib.request,
        "urlopen",
        lambda request, timeout=300: _Response(),
    )

    archives = bootstrap_windows._download_release_assets(
        assets=(asset,),
        stage_dir=stage_dir,
        progress_callback=None,
    )

    assert archives == ((stage_dir / asset.name).resolve(),)
    assert archives[0].read_bytes() == payload
