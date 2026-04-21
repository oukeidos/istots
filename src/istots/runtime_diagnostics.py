from __future__ import annotations

import atexit
import faulthandler
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

GUI_MANAGED_ROOT_ENV = "ISTOTS_GUI_MANAGED_ROOT"
RUNTIME_DIAGNOSTICS_FILENAME = "runtime-diagnostics.jsonl"
PYTHON_FAULTHANDLER_FILENAME = "python-faulthandler.log"
_MAX_LOG_BYTES = 1024 * 1024

_FAULTHANDLER_STREAM: TextIO | None = None
_FAULTHANDLER_REGISTERED = False


def diagnostics_state_dir() -> Path:
    configured_root = os.environ.get(GUI_MANAGED_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            root = Path(local_app_data).expanduser().resolve() / "istots" / "managed"
        else:
            root = (Path.home() / "AppData" / "Local" / "istots" / "managed").resolve()
    elif sys.platform == "darwin":
        root = (Path.home() / "Library" / "Application Support" / "istots" / "managed").resolve()
    else:
        root = (Path.home() / ".local" / "share" / "istots" / "managed").resolve()
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def diagnostics_log_path() -> Path:
    return diagnostics_state_dir() / RUNTIME_DIAGNOSTICS_FILENAME


def faulthandler_log_path() -> Path:
    return diagnostics_state_dir() / PYTHON_FAULTHANDLER_FILENAME


def append_runtime_diagnostic_event(event: str, /, **fields: Any) -> Path | None:
    try:
        path = diagnostics_log_path()
        _rotate_if_needed(path)
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
        }
        for key, value in fields.items():
            payload[key] = _json_safe(value)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return path
    except Exception:
        return None


def install_faulthandler_trace() -> Path | None:
    global _FAULTHANDLER_REGISTERED
    try:
        path = faulthandler_log_path()
        if _FAULTHANDLER_STREAM is not None and not _FAULTHANDLER_STREAM.closed:
            return path
        _rotate_if_needed(path)
        stream = path.open("a", encoding="utf-8")
        faulthandler.enable(file=stream, all_threads=True)
        _set_faulthandler_stream(stream)
        if not _FAULTHANDLER_REGISTERED:
            atexit.register(_close_faulthandler_stream)
            _FAULTHANDLER_REGISTERED = True
        append_runtime_diagnostic_event("python_faulthandler_enabled", log_path=path)
        return path
    except Exception:
        return None


def _set_faulthandler_stream(stream: TextIO) -> None:
    global _FAULTHANDLER_STREAM
    if _FAULTHANDLER_STREAM is not None and _FAULTHANDLER_STREAM is not stream:
        try:
            _FAULTHANDLER_STREAM.close()
        except Exception:
            pass
    _FAULTHANDLER_STREAM = stream


def _close_faulthandler_stream() -> None:
    global _FAULTHANDLER_STREAM
    if _FAULTHANDLER_STREAM is None:
        return
    try:
        faulthandler.disable()
    except Exception:
        pass
    try:
        _FAULTHANDLER_STREAM.close()
    except Exception:
        pass
    _FAULTHANDLER_STREAM = None


def _rotate_if_needed(path: Path) -> None:
    if not path.exists():
        return
    if path.stat().st_size < _MAX_LOG_BYTES:
        return
    archived = path.with_suffix(path.suffix + ".1")
    if archived.exists():
        archived.unlink()
    path.replace(archived)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value
