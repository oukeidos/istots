from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

_EXTERNAL_SUBPROCESS_LOCK = threading.Lock()


def sanitized_external_subprocess_env(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    if not getattr(sys, "frozen", False):
        return dict(base_env) if base_env is not None else None

    env = os.environ.copy()
    if base_env is not None:
        env.update(base_env)

    app_root = _frozen_application_root()
    if sys.platform == "win32":
        _sanitize_path_variable(env, "PATH", app_root)
        return env

    if sys.platform == "darwin":
        _sanitize_path_variable(env, "DYLD_LIBRARY_PATH", app_root)
        return env

    original_library_path = env.get("LD_LIBRARY_PATH_ORIG")
    if original_library_path is not None:
        if original_library_path:
            env["LD_LIBRARY_PATH"] = original_library_path
        else:
            env.pop("LD_LIBRARY_PATH", None)
        return env

    _sanitize_path_variable(env, "LD_LIBRARY_PATH", app_root)
    return env


@contextmanager
def sanitized_external_subprocess_runtime(
    base_env: Mapping[str, str] | None = None,
) -> Iterator[dict[str, str] | None]:
    env = sanitized_external_subprocess_env(base_env)
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        yield env
        return

    app_root = _frozen_application_root()
    if app_root is None:
        yield env
        return

    with _EXTERNAL_SUBPROCESS_LOCK:
        _set_windows_dll_directory(None)
        try:
            yield env
        finally:
            _set_windows_dll_directory(str(app_root))


def _sanitize_path_variable(
    env: dict[str, str],
    key: str,
    app_root: Path | None,
) -> None:
    if app_root is None:
        return
    raw_value = env.get(key, "")
    if not raw_value:
        return
    entries = [
        entry
        for entry in raw_value.split(os.pathsep)
        if entry and not _is_path_anchored_in_root(entry, app_root)
    ]
    if entries:
        env[key] = os.pathsep.join(entries)
    else:
        env.pop(key, None)


def _is_path_anchored_in_root(entry: str, root: Path) -> bool:
    try:
        normalized_entry = Path(entry).expanduser().resolve()
    except OSError:
        return False
    return normalized_entry == root or root in normalized_entry.parents


def _frozen_application_root() -> Path | None:
    raw_value = getattr(sys, "_MEIPASS", None)
    if not raw_value:
        return None
    return Path(str(raw_value)).resolve()


def _set_windows_dll_directory(path: str | None) -> None:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    result = kernel32.SetDllDirectoryW(path)
    if result == 0:
        raise OSError("SetDllDirectoryW failed while preparing an external subprocess launch")
