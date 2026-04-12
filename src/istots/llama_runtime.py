from __future__ import annotations

import base64
import fcntl
import io
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image

from istots.llama_mmproj import default_materialized_mmproj_path
from istots.llama_sampling import (
    PADDLEOCR_LLAMA_OCR_EXPLICIT_DEFAULTS,
    apply_openai_sampling_recipe,
)
from istots.model_store import (
    DEFAULT_GGUF_FILENAME,
    DEFAULT_GGUF_MODEL_ID,
    DEFAULT_GGUF_MMPROJ_FILENAME,
    resolve_local_model_path,
)

DEFAULT_LLAMA_SERVER_HOST = "127.0.0.1"
DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SEC = 120.0
DEFAULT_LLAMA_SERVER_SMOKE_MAX_TOKENS = 16
DEFAULT_LLAMA_SERVER_MANAGER_LOCK_PATH = Path("/tmp/istots-llama-server.lock")
DEFAULT_LLAMA_SERVER_MANAGER_STATE_PATH = Path("/tmp/istots-llama-server-state.json")

_ACTIVE_LLAMA_SERVER_MANAGER_LOCKS: dict[int, Any] = {}


class LlamaServerRole(StrEnum):
    OCR = "ocr"
    OCR_FAST = "ocr-fast"
    DETECTOR = "detector"
    CORRECTOR = "corrector"


class LlamaServerProfile(StrEnum):
    AUTO = "auto"
    CPU = "cpu"


DEFAULT_ROLE_PORTS: dict[LlamaServerRole, int] = {
    LlamaServerRole.OCR: 18080,
    LlamaServerRole.OCR_FAST: 18081,
    LlamaServerRole.DETECTOR: 18082,
    LlamaServerRole.CORRECTOR: 18083,
}


@dataclass(frozen=True)
class LlamaServerRoleAssets:
    role: LlamaServerRole
    model_path: Path
    mmproj_path: Path


@dataclass(frozen=True)
class LlamaServerOverrides:
    profile: LlamaServerProfile = LlamaServerProfile.AUTO
    threads: int | None = None
    threads_batch: int | None = None
    port: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool | None = None


@dataclass(frozen=True)
class LlamaServerLaunchSpec:
    role: LlamaServerRole
    profile: LlamaServerProfile
    binary_path: Path
    model_path: Path
    mmproj_path: Path
    host: str
    port: int
    threads: int | None = None
    threads_batch: int | None = None
    ctx_size: int | None = None
    n_predict: int | None = None
    reasoning: str | None = None
    reasoning_budget: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool = False
    prompt_text: str = "OCR:"


@dataclass(frozen=True)
class LlamaServerDoctorIssue:
    code: str
    message: str


@dataclass(frozen=True)
class LlamaServerDoctorReport:
    role: LlamaServerRole
    profile: LlamaServerProfile
    launch_spec: LlamaServerLaunchSpec | None
    issues: tuple[LlamaServerDoctorIssue, ...]
    smoke_response: str | None = None

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class LlamaServerManagerState:
    pid: int
    binary_path: str
    model_path: str
    mmproj_path: str
    host: str
    port: int
    role: str


def detect_llama_server_path(explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_path = os.environ.get("ISTOTS_LLAMA_SERVER_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    which = shutil_which("llama-server")
    if which is not None:
        candidates.append(Path(which))
    candidates.append(Path.home() / ".local" / "bin" / "llama-server")
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def shutil_which(binary: str) -> str | None:
    import shutil

    return shutil.which(binary)


def llama_server_manager_lock_path() -> Path:
    override = os.environ.get("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH")
    if override:
        return Path(override).expanduser()
    return DEFAULT_LLAMA_SERVER_MANAGER_LOCK_PATH


def llama_server_manager_state_path() -> Path:
    override = os.environ.get("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH")
    if override:
        return Path(override).expanduser()
    return DEFAULT_LLAMA_SERVER_MANAGER_STATE_PATH


def _acquire_llama_server_manager_lock() -> Any:
    lock_path = llama_server_manager_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_llama_server_manager_lock(handle: Any | None) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _load_llama_server_manager_state() -> LlamaServerManagerState | None:
    path = llama_server_manager_state_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LlamaServerManagerState(
        pid=int(payload["pid"]),
        binary_path=str(payload["binary_path"]),
        model_path=str(payload["model_path"]),
        mmproj_path=str(payload["mmproj_path"]),
        host=str(payload["host"]),
        port=int(payload["port"]),
        role=str(payload["role"]),
    )


def _write_llama_server_manager_state(
    spec: LlamaServerLaunchSpec,
    *,
    pid: int,
) -> None:
    path = llama_server_manager_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "binary_path": str(spec.binary_path),
        "model_path": str(spec.model_path),
        "mmproj_path": str(spec.mmproj_path),
        "host": spec.host,
        "port": spec.port,
        "role": spec.role.value,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _clear_llama_server_manager_state(*, pid: int | None = None) -> None:
    path = llama_server_manager_state_path()
    if not path.exists():
        return
    if pid is not None:
        state = _load_llama_server_manager_state()
        if state is not None and state.pid != pid:
            return
    path.unlink()


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_process_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _process_matches_manager_state(state: LlamaServerManagerState) -> bool:
    argv = _read_process_cmdline(state.pid)
    if not argv:
        return False
    try:
        binary_path = str(Path(argv[0]).expanduser().resolve())
    except OSError:
        binary_path = argv[0]
    return (
        binary_path == state.binary_path
        and state.model_path in argv
        and state.mmproj_path in argv
        and "--host" in argv
        and "--port" in argv
        and state.host in argv
        and str(state.port) in argv
    )


def _terminate_llama_server_pid(pid: int) -> None:
    if not _is_pid_alive(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return
        time.sleep(0.1)


def _cleanup_stale_llama_server_manager_state() -> None:
    state = _load_llama_server_manager_state()
    if state is None:
        return
    if not _is_pid_alive(state.pid):
        _clear_llama_server_manager_state()
        return
    if not _process_matches_manager_state(state):
        _clear_llama_server_manager_state()
        return
    _terminate_llama_server_pid(state.pid)
    _clear_llama_server_manager_state()


def resolve_llama_server_role_assets(
    role: str | LlamaServerRole,
    *,
    models_dir: Path | None = None,
    min_pixels: int = 32768,
) -> LlamaServerRoleAssets:
    normalized_role = normalize_llama_server_role(role)
    gguf_dir = resolve_local_model_path(DEFAULT_GGUF_MODEL_ID, models_dir=models_dir)
    model_path = (gguf_dir / DEFAULT_GGUF_FILENAME).resolve()
    base_mmproj_path = (gguf_dir / DEFAULT_GGUF_MMPROJ_FILENAME).resolve()
    fast_mmproj_path = default_materialized_mmproj_path(base_mmproj_path, min_pixels).resolve()

    if normalized_role is LlamaServerRole.OCR:
        return LlamaServerRoleAssets(
            role=normalized_role,
            model_path=model_path,
            mmproj_path=base_mmproj_path,
        )
    if normalized_role is LlamaServerRole.OCR_FAST:
        return LlamaServerRoleAssets(
            role=normalized_role,
            model_path=model_path,
            mmproj_path=fast_mmproj_path,
        )
    if normalized_role is LlamaServerRole.DETECTOR:
        return LlamaServerRoleAssets(
            role=normalized_role,
            model_path=model_path,
            mmproj_path=base_mmproj_path,
        )
    raise RuntimeError(
        "The corrector runtime assets are not provisioned by core setup yet. "
        "Run doctor for OCR-oriented roles until correction provisioning is implemented."
    )


def build_llama_server_launch_spec(
    *,
    role: str | LlamaServerRole,
    binary_path: Path,
    models_dir: Path | None = None,
    min_pixels: int = 32768,
    host: str = DEFAULT_LLAMA_SERVER_HOST,
    overrides: LlamaServerOverrides | None = None,
) -> LlamaServerLaunchSpec:
    normalized_role = normalize_llama_server_role(role)
    normalized_overrides = overrides or LlamaServerOverrides()
    assets = resolve_llama_server_role_assets(
        normalized_role,
        models_dir=models_dir,
        min_pixels=min_pixels,
    )
    return LlamaServerLaunchSpec(
        role=normalized_role,
        profile=normalized_overrides.profile,
        binary_path=binary_path.expanduser().resolve(),
        model_path=assets.model_path,
        mmproj_path=assets.mmproj_path,
        host=host,
        port=normalized_overrides.port or DEFAULT_ROLE_PORTS[normalized_role],
        threads=normalized_overrides.threads,
        threads_batch=normalized_overrides.threads_batch,
        gpu_layers=normalized_overrides.gpu_layers,
        no_mmproj_offload=bool(normalized_overrides.no_mmproj_offload),
    )


def build_llama_server_command(spec: LlamaServerLaunchSpec) -> list[str]:
    command = [
        str(spec.binary_path),
        "-m",
        str(spec.model_path),
        "--mmproj",
        str(spec.mmproj_path),
        "--host",
        spec.host,
        "--port",
        str(spec.port),
    ]

    force_cpu = spec.profile is LlamaServerProfile.CPU
    no_mmproj_offload = bool(spec.no_mmproj_offload)

    if force_cpu:
        command.extend(["--device", "none", "--gpu-layers", "0"])
    elif spec.gpu_layers is not None:
        command.extend(["--gpu-layers", str(spec.gpu_layers)])

    if spec.threads is not None:
        command.extend(["-t", str(spec.threads)])
    if spec.threads_batch is not None:
        command.extend(["-tb", str(spec.threads_batch)])
    if spec.ctx_size is not None:
        command.extend(["-c", str(spec.ctx_size)])
    if spec.n_predict is not None:
        command.extend(["-n", str(spec.n_predict)])
    if spec.reasoning is not None:
        command.extend(["--reasoning", str(spec.reasoning)])
    if spec.reasoning_budget is not None:
        command.extend(["--reasoning-budget", str(spec.reasoning_budget)])
    if no_mmproj_offload:
        command.append("--no-mmproj-offload")

    return command


def wait_until_ready(
    host: str,
    port: int,
    timeout_sec: float,
    *,
    process: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_sec
    url = f"http://{host}:{port}/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited before becoming ready (exit={process.returncode})"
            )
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - exercised through timeout paths
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(
        f"llama-server did not become ready at {url} within {timeout_sec} seconds"
    ) from last_error


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _reserved_llama_server_ports(target_port: int) -> list[int]:
    return sorted({*DEFAULT_ROLE_PORTS.values(), target_port})


def start_llama_server(
    spec: LlamaServerLaunchSpec,
    startup_timeout_sec: float,
) -> subprocess.Popen[str]:
    if _ACTIVE_LLAMA_SERVER_MANAGER_LOCKS:
        raise RuntimeError(
            "llama-server manager already has an active runtime in this process; "
            "close the current runtime before starting another one."
        )

    manager_lock = _acquire_llama_server_manager_lock()
    command = build_llama_server_command(spec)
    try:
        _cleanup_stale_llama_server_manager_state()
        occupied_ports = [port for port in _reserved_llama_server_ports(spec.port) if is_port_in_use(spec.host, port)]
        if occupied_ports:
            joined = ", ".join(f"{spec.host}:{port}" for port in occupied_ports)
            raise RuntimeError(f"reserved llama-server ports are already in use: {joined}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        _write_llama_server_manager_state(spec, pid=process.pid)
        _ACTIVE_LLAMA_SERVER_MANAGER_LOCKS[process.pid] = manager_lock
        wait_until_ready(
            spec.host,
            spec.port,
            timeout_sec=startup_timeout_sec,
            process=process,
        )
        return process
    except Exception:
        if "process" in locals():
            stop_llama_server(process)
        else:
            _clear_llama_server_manager_state()
            _release_llama_server_manager_lock(manager_lock)
        raise


def stop_llama_server(process: subprocess.Popen[str]) -> None:
    manager_lock = _ACTIVE_LLAMA_SERVER_MANAGER_LOCKS.pop(process.pid, None)
    try:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    finally:
        _clear_llama_server_manager_state(pid=process.pid)
        _release_llama_server_manager_lock(manager_lock)


def request_llama_server_smoke(spec: LlamaServerLaunchSpec) -> str:
    image = Image.new("RGB", (1, 1), "white")
    return request_llama_server_ocr(
        spec,
        image,
        max_new_tokens=DEFAULT_LLAMA_SERVER_SMOKE_MAX_TOKENS,
        prompt_text=spec.prompt_text,
    )


def request_llama_server_ocr(
    spec: LlamaServerLaunchSpec,
    image: Image.Image,
    *,
    max_new_tokens: int,
    prompt_text: str = "OCR:",
) -> str:
    url = f"http://{spec.host}:{spec.port}/v1/chat/completions"
    body: dict[str, Any] = {
        "model": "gpt-3.5-turbo",
        "max_tokens": max_new_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                ],
            }
        ],
    }
    apply_openai_sampling_recipe(
        body,
        recipe=PADDLEOCR_LLAMA_OCR_EXPLICIT_DEFAULTS,
    )
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    return str(parsed["choices"][0]["message"]["content"]).strip()


def run_llama_server_launch_spec_doctor(
    spec: LlamaServerLaunchSpec,
    *,
    startup_timeout_sec: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SEC,
) -> LlamaServerDoctorReport:
    issues: list[LlamaServerDoctorIssue] = []

    for code, path in (
        ("missing_binary", spec.binary_path),
        ("missing_model", spec.model_path),
        ("missing_mmproj", spec.mmproj_path),
    ):
        if not path.exists():
            issues.append(
                LlamaServerDoctorIssue(
                    code=code,
                    message=f"required runtime asset is missing: {path}",
                )
            )

    manager_lock = _acquire_llama_server_manager_lock()
    try:
        _cleanup_stale_llama_server_manager_state()
        try:
            port_in_use = is_port_in_use(spec.host, spec.port)
        except OSError as exc:
            issues.append(
                LlamaServerDoctorIssue(
                    code="port_probe_failed",
                    message=f"failed to probe port readiness: {exc}",
                )
            )
            port_in_use = False
    finally:
        _release_llama_server_manager_lock(manager_lock)

    if port_in_use:
        issues.append(
            LlamaServerDoctorIssue(
                code="port_in_use",
                message=f"requested port is already in use: {spec.host}:{spec.port}",
            )
        )

    if issues:
        return LlamaServerDoctorReport(
            role=spec.role,
            profile=spec.profile,
            launch_spec=spec,
            issues=tuple(issues),
        )

    process = start_llama_server(spec, startup_timeout_sec=startup_timeout_sec)
    try:
        smoke_response = request_llama_server_smoke(spec)
    except Exception as exc:
        issues.append(
            LlamaServerDoctorIssue(
                code="smoke_failed",
                message=str(exc),
            )
        )
        return LlamaServerDoctorReport(
            role=spec.role,
            profile=spec.profile,
            launch_spec=spec,
            issues=tuple(issues),
        )
    finally:
        stop_llama_server(process)

    return LlamaServerDoctorReport(
        role=spec.role,
        profile=spec.profile,
        launch_spec=spec,
        issues=tuple(),
        smoke_response=smoke_response,
    )


def run_llama_server_doctor(
    *,
    role: str | LlamaServerRole,
    models_dir: Path | None = None,
    min_pixels: int = 32768,
    explicit_binary_path: Path | None = None,
    host: str = DEFAULT_LLAMA_SERVER_HOST,
    overrides: LlamaServerOverrides | None = None,
    startup_timeout_sec: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SEC,
) -> LlamaServerDoctorReport:
    normalized_role = normalize_llama_server_role(role)
    normalized_overrides = overrides or LlamaServerOverrides()
    issues: list[LlamaServerDoctorIssue] = []

    binary_path = detect_llama_server_path(explicit_binary_path)
    if binary_path is None:
        issues.append(
            LlamaServerDoctorIssue(
                code="missing_binary",
                message="llama-server binary not found. Set ISTOTS_LLAMA_SERVER_PATH or pass --llama-server-path.",
            )
        )
        return LlamaServerDoctorReport(
            role=normalized_role,
            profile=normalized_overrides.profile,
            launch_spec=None,
            issues=tuple(issues),
        )

    try:
        launch_spec = build_llama_server_launch_spec(
            role=normalized_role,
            binary_path=binary_path,
            models_dir=models_dir,
            min_pixels=min_pixels,
            host=host,
            overrides=normalized_overrides,
        )
    except Exception as exc:
        issues.append(
            LlamaServerDoctorIssue(
                code="asset_resolution_failed",
                message=str(exc),
            )
        )
        return LlamaServerDoctorReport(
            role=normalized_role,
            profile=normalized_overrides.profile,
            launch_spec=None,
            issues=tuple(issues),
        )

    return run_llama_server_launch_spec_doctor(
        launch_spec,
        startup_timeout_sec=startup_timeout_sec,
    )


def normalize_llama_server_role(role: str | LlamaServerRole) -> LlamaServerRole:
    if isinstance(role, LlamaServerRole):
        return role
    return LlamaServerRole(role)


def normalize_llama_server_profile(profile: str | LlamaServerProfile) -> LlamaServerProfile:
    if isinstance(profile, LlamaServerProfile):
        return profile
    return LlamaServerProfile(profile)


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"
