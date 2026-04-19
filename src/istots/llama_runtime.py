from __future__ import annotations

import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
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
DEFAULT_LLAMA_SERVER_MANAGER_DIRNAME = "llama-server-manager"
DEFAULT_LLAMA_SERVER_MANAGER_LOCK_DIRNAME = "lock"
DEFAULT_LLAMA_SERVER_MANAGER_LOCK_OWNER_FILENAME = "owner.json"
DEFAULT_LLAMA_SERVER_MANAGER_STATE_FILENAME = "state.json"
DEFAULT_LLAMA_SERVER_MANAGER_LOCK_POLL_SEC = 0.1
DEFAULT_LLAMA_SERVER_MANAGER_LOCK_STALE_GRACE_SEC = 5.0

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
    ctx_size: int | None = None
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
    connect_host: str | None = None
    threads: int | None = None
    threads_batch: int | None = None
    ctx_size: int | None = None
    n_predict: int | None = None
    reasoning: str | None = None
    reasoning_budget: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool = False
    prompt_text: str = "OCR:"

    def __post_init__(self) -> None:
        if self.connect_host is None:
            object.__setattr__(self, "connect_host", derive_llama_server_connect_host(self.host))


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
    instance_id: str
    created_at: float
    pid: int
    binary_path: str
    model_path: str
    mmproj_path: str
    bind_host: str
    connect_host: str
    port: int
    role: str


@dataclass(frozen=True)
class LlamaServerManagerPaths:
    runtime_root: Path
    manager_dir: Path
    lock_dir: Path
    lock_owner_path: Path
    state_path: Path


@dataclass(frozen=True)
class LlamaServerManagerLockOwner:
    pid: int
    instance_id: str
    created_at: float


@dataclass(frozen=True)
class LlamaServerManagerLock:
    paths: LlamaServerManagerPaths
    owner: LlamaServerManagerLockOwner


@dataclass(frozen=True)
class LlamaServerOCRResponse:
    text: str
    finish_reason: str | None = None
    completion_tokens: int | None = None


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


def llama_server_manager_runtime_root() -> Path:
    runtime_override = os.environ.get("ISTOTS_LLAMA_SERVER_MANAGER_RUNTIME_DIR")
    if runtime_override:
        return Path(runtime_override).expanduser()

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).expanduser() / "istots" / "runtime"
        return Path.home() / "AppData" / "Local" / "istots" / "runtime"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "istots" / "runtime"

    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        return Path(xdg_runtime_dir).expanduser() / "istots"
    return Path.home() / ".local" / "state" / "istots" / "runtime"


def llama_server_manager_paths() -> LlamaServerManagerPaths:
    runtime_root = llama_server_manager_runtime_root()
    manager_dir = runtime_root / DEFAULT_LLAMA_SERVER_MANAGER_DIRNAME

    lock_override = os.environ.get("ISTOTS_LLAMA_SERVER_MANAGER_LOCK_PATH")
    if lock_override:
        lock_dir = Path(lock_override).expanduser()
    else:
        lock_dir = manager_dir / DEFAULT_LLAMA_SERVER_MANAGER_LOCK_DIRNAME

    state_override = os.environ.get("ISTOTS_LLAMA_SERVER_MANAGER_STATE_PATH")
    if state_override:
        state_path = Path(state_override).expanduser()
    else:
        state_path = manager_dir / DEFAULT_LLAMA_SERVER_MANAGER_STATE_FILENAME

    return LlamaServerManagerPaths(
        runtime_root=runtime_root,
        manager_dir=manager_dir,
        lock_dir=lock_dir,
        lock_owner_path=lock_dir / DEFAULT_LLAMA_SERVER_MANAGER_LOCK_OWNER_FILENAME,
        state_path=state_path,
    )


def llama_server_manager_lock_path() -> Path:
    return llama_server_manager_paths().lock_dir


def llama_server_manager_state_path() -> Path:
    return llama_server_manager_paths().state_path


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass
    temp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _load_manager_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_llama_server_manager_lock_owner(paths: LlamaServerManagerPaths | None = None) -> LlamaServerManagerLockOwner | None:
    normalized_paths = paths or llama_server_manager_paths()
    payload = _load_manager_json(normalized_paths.lock_owner_path)
    if payload is None:
        return None
    try:
        return LlamaServerManagerLockOwner(
            pid=int(payload["pid"]),
            instance_id=str(payload["instance_id"]),
            created_at=float(payload["created_at"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_llama_server_manager_lock_owner(lock: LlamaServerManagerLock) -> None:
    _atomic_write_json(
        lock.paths.lock_owner_path,
        {
            "pid": lock.owner.pid,
            "instance_id": lock.owner.instance_id,
            "created_at": lock.owner.created_at,
        },
    )


def _remove_llama_server_manager_lock_dir(paths: LlamaServerManagerPaths) -> None:
    try:
        paths.lock_owner_path.unlink()
    except FileNotFoundError:
        pass
    try:
        paths.lock_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        # Only manager metadata lives in this directory; fall back to clearing
        # best-effort if a stale partial startup left extra files behind.
        for child in paths.lock_dir.iterdir():
            if child.is_dir():
                continue
            try:
                child.unlink()
            except OSError:
                return
        try:
            paths.lock_dir.rmdir()
        except OSError:
            return


def _cleanup_stale_llama_server_manager_lock(paths: LlamaServerManagerPaths) -> bool:
    if not paths.lock_dir.exists():
        return False
    owner = _load_llama_server_manager_lock_owner(paths)
    if owner is not None:
        if _is_pid_alive(owner.pid):
            return False
        _clear_llama_server_manager_state(instance_id=owner.instance_id)
        _remove_llama_server_manager_lock_dir(paths)
        return True
    try:
        age_sec = time.time() - paths.lock_dir.stat().st_mtime
    except OSError:
        return False
    if age_sec < DEFAULT_LLAMA_SERVER_MANAGER_LOCK_STALE_GRACE_SEC:
        return False
    _remove_llama_server_manager_lock_dir(paths)
    return True


def _acquire_llama_server_manager_lock() -> LlamaServerManagerLock:
    paths = llama_server_manager_paths()
    _ensure_private_directory(paths.runtime_root)
    _ensure_private_directory(paths.manager_dir)
    _ensure_private_directory(paths.lock_dir.parent)
    owner = LlamaServerManagerLockOwner(
        pid=os.getpid(),
        instance_id=uuid.uuid4().hex,
        created_at=time.time(),
    )
    while True:
        try:
            paths.lock_dir.mkdir(mode=0o700)
        except FileExistsError:
            if _cleanup_stale_llama_server_manager_lock(paths):
                continue
            time.sleep(DEFAULT_LLAMA_SERVER_MANAGER_LOCK_POLL_SEC)
            continue
        try:
            os.chmod(paths.lock_dir, 0o700)
        except OSError:
            pass
        lock = LlamaServerManagerLock(paths=paths, owner=owner)
        _write_llama_server_manager_lock_owner(lock)
        return lock


def _release_llama_server_manager_lock(lock: LlamaServerManagerLock | None) -> None:
    if lock is None:
        return
    current_owner = _load_llama_server_manager_lock_owner(lock.paths)
    if current_owner is not None and current_owner.instance_id != lock.owner.instance_id:
        return
    _remove_llama_server_manager_lock_dir(lock.paths)


def _load_llama_server_manager_state() -> LlamaServerManagerState | None:
    path = llama_server_manager_state_path()
    payload = _load_manager_json(path)
    if payload is None:
        return None
    bind_host = str(payload.get("bind_host", payload.get("host", DEFAULT_LLAMA_SERVER_HOST)))
    return LlamaServerManagerState(
        instance_id=str(payload.get("instance_id", "")),
        created_at=float(payload.get("created_at", 0.0)),
        pid=int(payload["pid"]),
        binary_path=str(payload["binary_path"]),
        model_path=str(payload["model_path"]),
        mmproj_path=str(payload["mmproj_path"]),
        bind_host=bind_host,
        connect_host=str(payload.get("connect_host", derive_llama_server_connect_host(bind_host))),
        port=int(payload["port"]),
        role=str(payload["role"]),
    )


def _write_llama_server_manager_state(
    spec: LlamaServerLaunchSpec,
    *,
    pid: int,
    instance_id: str,
) -> None:
    path = llama_server_manager_state_path()
    payload = {
        "instance_id": instance_id,
        "created_at": time.time(),
        "pid": pid,
        "binary_path": str(spec.binary_path),
        "model_path": str(spec.model_path),
        "mmproj_path": str(spec.mmproj_path),
        "bind_host": spec.host,
        "connect_host": spec.connect_host,
        "port": spec.port,
        "role": spec.role.value,
    }
    _atomic_write_json(path, payload)


def _clear_llama_server_manager_state(
    *,
    pid: int | None = None,
    instance_id: str | None = None,
) -> None:
    path = llama_server_manager_state_path()
    if not path.exists():
        return
    if pid is not None or instance_id is not None:
        state = _load_llama_server_manager_state()
        if state is None:
            return
        if pid is not None and state.pid != pid:
            return
        if instance_id is not None and state.instance_id != instance_id:
            return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_llama_server_manager_state() -> None:
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


def derive_llama_server_connect_host(bind_host: str) -> str:
    normalized = bind_host.strip()
    if normalized in {"", "0.0.0.0"}:
        return DEFAULT_LLAMA_SERVER_HOST
    if normalized in {"::", "[::]"}:
        return "::1"
    return bind_host


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
    normalized_profile = normalize_llama_server_profile(normalized_overrides.profile)
    assets = resolve_llama_server_role_assets(
        normalized_role,
        models_dir=models_dir,
        min_pixels=min_pixels,
    )
    return LlamaServerLaunchSpec(
        role=normalized_role,
        profile=normalized_profile,
        binary_path=binary_path.expanduser().resolve(),
        model_path=assets.model_path,
        mmproj_path=assets.mmproj_path,
        host=host,
        port=normalized_overrides.port or DEFAULT_ROLE_PORTS[normalized_role],
        threads=normalized_overrides.threads,
        threads_batch=normalized_overrides.threads_batch,
        ctx_size=normalized_overrides.ctx_size,
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


def _terminate_started_llama_server_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
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
        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "text": True,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **popen_kwargs)
        _write_llama_server_manager_state(
            spec,
            pid=process.pid,
            instance_id=manager_lock.owner.instance_id,
        )
        _ACTIVE_LLAMA_SERVER_MANAGER_LOCKS[process.pid] = manager_lock
        wait_until_ready(
            spec.connect_host,
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
        _terminate_started_llama_server_process(process)
    finally:
        instance_id = manager_lock.owner.instance_id if manager_lock is not None else None
        _clear_llama_server_manager_state(pid=process.pid, instance_id=instance_id)
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
    return request_llama_server_ocr_response(
        spec,
        image,
        max_new_tokens=max_new_tokens,
        prompt_text=prompt_text,
    ).text


def request_llama_server_ocr_response(
    spec: LlamaServerLaunchSpec,
    image: Image.Image,
    *,
    max_new_tokens: int,
    prompt_text: str = "OCR:",
) -> LlamaServerOCRResponse:
    url = f"http://{spec.connect_host}:{spec.port}/v1/chat/completions"
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
    return LlamaServerOCRResponse(
        text=str(parsed["choices"][0]["message"]["content"]).strip(),
        finish_reason=parsed["choices"][0].get("finish_reason"),
        completion_tokens=parsed.get("usage", {}).get("completion_tokens"),
    )


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
    normalized_profile = normalize_llama_server_profile(normalized_overrides.profile)
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
            profile=normalized_profile,
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
            profile=normalized_profile,
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
