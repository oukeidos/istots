from __future__ import annotations

import json
import stat
from pathlib import Path

from istots import llama_runtime


def test_build_llama_server_command_uses_auto_profile_defaults() -> None:
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18080,
    )

    assert llama_runtime.build_llama_server_command(spec) == [
        "/tmp/llama-server",
        "-m",
        "/tmp/model.gguf",
        "--mmproj",
        "/tmp/mmproj.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
    ]


def test_llama_server_launch_spec_derives_loopback_connect_host_for_wildcard_bind() -> None:
    ipv4_spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="0.0.0.0",
        port=18080,
    )
    ipv6_spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="::",
        port=18080,
    )

    assert ipv4_spec.connect_host == "127.0.0.1"
    assert ipv6_spec.connect_host == "::1"


def test_build_llama_server_command_applies_cpu_profile_and_overrides() -> None:
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.CPU,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18080,
        threads=12,
        threads_batch=8,
    )

    assert llama_runtime.build_llama_server_command(spec) == [
        "/tmp/llama-server",
        "-m",
        "/tmp/model.gguf",
        "--mmproj",
        "/tmp/mmproj.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "-t",
        "12",
        "-tb",
        "8",
    ]

def test_build_llama_server_command_applies_explicit_no_mmproj_offload_override() -> None:
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.CORRECTOR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18083,
        no_mmproj_offload=True,
    )

    assert llama_runtime.build_llama_server_command(spec) == [
        "/tmp/llama-server",
        "-m",
        "/tmp/model.gguf",
        "--mmproj",
        "/tmp/mmproj.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "18083",
        "--no-mmproj-offload",
    ]


def test_build_llama_server_command_appends_context_and_reasoning_flags() -> None:
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.CORRECTOR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18083,
        ctx_size=4096,
        n_predict=128,
        reasoning="off",
    )

    assert llama_runtime.build_llama_server_command(spec) == [
        "/tmp/llama-server",
        "-m",
        "/tmp/model.gguf",
        "--mmproj",
        "/tmp/mmproj.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "18083",
        "-c",
        "4096",
        "-n",
        "128",
        "--reasoning",
        "off",
    ]


def test_resolve_llama_server_role_assets_uses_derived_mmproj_for_fast_role(monkeypatch, tmp_path: Path) -> None:
    gguf_dir = tmp_path / "PaddlePaddle__PaddleOCR-VL-1.5-GGUF"
    monkeypatch.setattr(llama_runtime, "resolve_local_model_path", lambda model_id, models_dir=None: gguf_dir)

    assets = llama_runtime.resolve_llama_server_role_assets(
        llama_runtime.LlamaServerRole.OCR_FAST,
        models_dir=tmp_path,
        min_pixels=32768,
    )

    assert assets.model_path == (gguf_dir / "PaddleOCR-VL-1.5.gguf").resolve()
    assert assets.mmproj_path == (gguf_dir / "PaddleOCR-VL-1.5-mmproj.minpix32768.gguf").resolve()


def test_llama_server_manager_paths_use_runtime_dir_override(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime-root"
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_RUNTIME_DIR", str(runtime_root))

    paths = llama_runtime.llama_server_manager_paths()

    assert paths.runtime_root == runtime_root
    assert paths.manager_dir == runtime_root / "llama-server-manager"
    assert paths.lock_dir == runtime_root / "llama-server-manager" / "lock"
    assert paths.state_path == runtime_root / "llama-server-manager" / "state.json"


def test_run_llama_server_doctor_reports_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(llama_runtime, "detect_llama_server_path", lambda explicit=None: None)
    report = llama_runtime.run_llama_server_doctor(role="ocr")
    assert report.ok is False
    assert report.issues[0].code == "missing_binary"


def test_run_llama_server_doctor_runs_smoke_on_ready_runtime(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    process = object()
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH", str(tmp_path / "llama.lock"))
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(tmp_path / "llama-state.json"))

    monkeypatch.setattr(llama_runtime, "detect_llama_server_path", lambda explicit=None: binary)
    monkeypatch.setattr(
        llama_runtime,
        "build_llama_server_launch_spec",
        lambda **kwargs: llama_runtime.LlamaServerLaunchSpec(
            role=llama_runtime.LlamaServerRole.OCR,
            profile=llama_runtime.LlamaServerProfile.AUTO,
            binary_path=binary,
            model_path=model,
            mmproj_path=mmproj,
            host="127.0.0.1",
            port=18080,
        ),
    )
    monkeypatch.setattr(llama_runtime, "is_port_in_use", lambda host, port: False)
    monkeypatch.setattr(llama_runtime, "start_llama_server", lambda spec, startup_timeout_sec: process)
    monkeypatch.setattr(llama_runtime, "request_llama_server_smoke", lambda spec: "OK")

    stopped: list[object] = []
    monkeypatch.setattr(llama_runtime, "stop_llama_server", lambda proc: stopped.append(proc))

    report = llama_runtime.run_llama_server_doctor(role="ocr")

    assert report.ok is True
    assert report.smoke_response == "OK"
    assert stopped == [process]


def test_run_llama_server_launch_spec_doctor_runs_smoke_on_ready_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    process = object()
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH", str(tmp_path / "llama.lock"))
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(tmp_path / "llama-state.json"))

    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.CORRECTOR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=binary,
        model_path=model,
        mmproj_path=mmproj,
        host="127.0.0.1",
        port=18083,
        prompt_text="STRICT",
    )

    monkeypatch.setattr(llama_runtime, "is_port_in_use", lambda host, port: False)
    monkeypatch.setattr(llama_runtime, "start_llama_server", lambda spec, startup_timeout_sec: process)
    monkeypatch.setattr(llama_runtime, "request_llama_server_smoke", lambda spec: "STRICT-OK")

    stopped: list[object] = []
    monkeypatch.setattr(llama_runtime, "stop_llama_server", lambda proc: stopped.append(proc))

    report = llama_runtime.run_llama_server_launch_spec_doctor(spec)

    assert report.ok is True
    assert report.role is llama_runtime.LlamaServerRole.CORRECTOR
    assert report.smoke_response == "STRICT-OK"
    assert stopped == [process]


class _FakePopen:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


def test_start_llama_server_reclaims_stale_dead_owner_lock_before_launch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "llama.lock"
    state_path = tmp_path / "llama-state.json"
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18080,
    )
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(
        json.dumps(
            {
                "pid": 4321,
                "instance_id": "stale-owner",
                "created_at": 10.0,
            }
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "instance_id": "stale-owner",
                "created_at": 10.0,
                "pid": 4321,
                "binary_path": "/tmp/llama-server",
                "model_path": "/tmp/old-model.gguf",
                "mmproj_path": "/tmp/old-mmproj.gguf",
                "host": "127.0.0.1",
                "port": 18080,
                "role": "ocr",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(state_path))
    monkeypatch.setattr(llama_runtime, "_ACTIVE_LLAMA_SERVER_MANAGER_LOCKS", {})
    monkeypatch.setattr(llama_runtime, "build_llama_server_command", lambda spec: ["llama-server"])
    monkeypatch.setattr(llama_runtime, "_is_pid_alive", lambda pid: False)
    monkeypatch.setattr(llama_runtime, "is_port_in_use", lambda host, port: False)
    monkeypatch.setattr(llama_runtime, "wait_until_ready", lambda host, port, timeout_sec, process=None: None)
    monkeypatch.setattr(
        llama_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakePopen(pid=9876),
    )

    process = llama_runtime.start_llama_server(spec, startup_timeout_sec=1.0)
    try:
        assert process.pid == 9876
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert payload["pid"] == 9876
        assert payload["model_path"] == "/tmp/model.gguf"
        assert payload["bind_host"] == "127.0.0.1"
        assert payload["connect_host"] == "127.0.0.1"
        assert payload["instance_id"] != "stale-owner"
    finally:
        llama_runtime.stop_llama_server(process)


def test_start_llama_server_waits_on_connect_host(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "llama.lock"
    state_path = tmp_path / "llama-state.json"
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="0.0.0.0",
        port=18080,
    )
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(state_path))
    monkeypatch.setattr(llama_runtime, "_ACTIVE_LLAMA_SERVER_MANAGER_LOCKS", {})
    monkeypatch.setattr(llama_runtime, "build_llama_server_command", lambda spec: ["llama-server"])
    monkeypatch.setattr(llama_runtime, "is_port_in_use", lambda host, port: False)
    monkeypatch.setattr(
        llama_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakePopen(pid=1234),
    )

    waited_on: list[tuple[str, int]] = []
    monkeypatch.setattr(
        llama_runtime,
        "wait_until_ready",
        lambda host, port, timeout_sec, process=None: waited_on.append((host, port)),
    )

    process = llama_runtime.start_llama_server(spec, startup_timeout_sec=1.0)
    try:
        assert waited_on == [("127.0.0.1", 18080)]
    finally:
        llama_runtime.stop_llama_server(process)


def test_stop_llama_server_clears_manager_state(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "llama.lock"
    state_path = tmp_path / "llama-state.json"
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18080,
    )
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(state_path))
    monkeypatch.setattr(llama_runtime, "_ACTIVE_LLAMA_SERVER_MANAGER_LOCKS", {})
    monkeypatch.setattr(llama_runtime, "build_llama_server_command", lambda spec: ["llama-server"])
    monkeypatch.setattr(llama_runtime, "is_port_in_use", lambda host, port: False)
    monkeypatch.setattr(llama_runtime, "wait_until_ready", lambda host, port, timeout_sec, process=None: None)
    monkeypatch.setattr(
        llama_runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakePopen(pid=2468),
    )

    terminated: list[int] = []
    monkeypatch.setattr(
        llama_runtime,
        "_terminate_started_llama_server_process",
        lambda process: terminated.append(process.pid),
    )

    process = llama_runtime.start_llama_server(spec, startup_timeout_sec=1.0)
    assert state_path.exists()
    assert lock_path.exists()

    llama_runtime.stop_llama_server(process)

    assert terminated == [2468]
    assert state_path.exists() is False
    assert lock_path.exists() is False
    assert llama_runtime._ACTIVE_LLAMA_SERVER_MANAGER_LOCKS == {}


def test_load_llama_server_manager_state_accepts_legacy_host(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "llama-state.json"
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(state_path))
    state_path.write_text(
        json.dumps(
            {
                "instance_id": "legacy",
                "created_at": 10.0,
                "pid": 4321,
                "binary_path": "/tmp/llama-server",
                "model_path": "/tmp/model.gguf",
                "mmproj_path": "/tmp/mmproj.gguf",
                "host": "0.0.0.0",
                "port": 18080,
                "role": "ocr",
            }
        ),
        encoding="utf-8",
    )

    state = llama_runtime._load_llama_server_manager_state()

    assert state is not None
    assert state.bind_host == "0.0.0.0"
    assert state.connect_host == "127.0.0.1"


def test_start_llama_server_rejects_occupied_reserved_ports(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "llama.lock"
    state_path = tmp_path / "llama-state.json"
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.CORRECTOR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="127.0.0.1",
        port=18083,
    )
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(state_path))
    monkeypatch.setattr(llama_runtime, "_ACTIVE_LLAMA_SERVER_MANAGER_LOCKS", {})
    monkeypatch.setattr(llama_runtime, "build_llama_server_command", lambda spec: ["llama-server"])
    monkeypatch.setattr(
        llama_runtime,
        "is_port_in_use",
        lambda host, port: port in {18080, 18083},
    )

    try:
        llama_runtime.start_llama_server(spec, startup_timeout_sec=1.0)
    except RuntimeError as exc:
        assert str(exc) == "reserved llama-server ports are already in use: 127.0.0.1:18080, 127.0.0.1:18083"
    else:
        raise AssertionError("expected reserved-port conflict")


def test_manager_lock_and_state_use_private_permissions(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "llama.lock"
    state_path = tmp_path / "llama-state.json"
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH", str(state_path))

    lock = llama_runtime._acquire_llama_server_manager_lock()
    try:
        spec = llama_runtime.LlamaServerLaunchSpec(
            role=llama_runtime.LlamaServerRole.OCR,
            profile=llama_runtime.LlamaServerProfile.AUTO,
            binary_path=Path("/tmp/llama-server"),
            model_path=Path("/tmp/model.gguf"),
            mmproj_path=Path("/tmp/mmproj.gguf"),
            host="127.0.0.1",
            port=18080,
        )
        llama_runtime._write_llama_server_manager_state(
            spec,
            pid=1234,
            instance_id=lock.owner.instance_id,
        )

        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o700
        assert stat.S_IMODE((lock_path / "owner.json").stat().st_mode) == 0o600
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    finally:
        llama_runtime._clear_llama_server_manager_state(instance_id=lock.owner.instance_id)
        llama_runtime._release_llama_server_manager_lock(lock)


def test_request_llama_server_ocr_response_uses_connect_host(monkeypatch) -> None:
    spec = llama_runtime.LlamaServerLaunchSpec(
        role=llama_runtime.LlamaServerRole.OCR,
        profile=llama_runtime.LlamaServerProfile.AUTO,
        binary_path=Path("/tmp/llama-server"),
        model_path=Path("/tmp/model.gguf"),
        mmproj_path=Path("/tmp/mmproj.gguf"),
        host="0.0.0.0",
        port=18080,
    )

    captured_urls: list[str] = []

    class _FakeResponse:
        status = 200

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout=60):
        captured_urls.append(request.full_url)
        return _FakeResponse()

    monkeypatch.setattr(llama_runtime.urllib.request, "urlopen", fake_urlopen)

    response = llama_runtime.request_llama_server_ocr_response(
        spec,
        llama_runtime.Image.new("RGB", (1, 1), "white"),
        max_new_tokens=8,
    )

    assert captured_urls == ["http://127.0.0.1:18080/v1/chat/completions"]
    assert response.text == "ok"
