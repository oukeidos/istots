from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GEMINI_AUTH_CONFIG_PATH = Path.home() / ".config" / "istots" / "auth.json"
GEMINI_KEYRING_SERVICE = "istots"
GEMINI_KEYRING_USERNAME = "gemini-api-key"


@dataclass(frozen=True)
class GeminiAuthStatus:
    keyring_backend: str | None
    keyring_available: bool
    keyring_configured: bool
    env_file_path: Path | None
    env_file_exists: bool
    env_file_configured: bool
    env_file_contains_key: bool
    process_env_name: str | None
    process_env_configured: bool
    effective_source: str | None


def gemini_auth_config_path() -> Path:
    configured = os.environ.get("ISTOTS_AUTH_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_GEMINI_AUTH_CONFIG_PATH


def _load_auth_config() -> dict[str, str]:
    path = gemini_auth_config_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_auth_config(payload: dict[str, str]) -> None:
    path = gemini_auth_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_configured_gemini_env_file() -> Path | None:
    payload = _load_auth_config()
    raw = payload.get("gemini_env_file_path")
    if not raw:
        return None
    return Path(raw).expanduser()


def set_configured_gemini_env_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise RuntimeError(f"Gemini .env path must be a file path: {resolved}")
    _write_auth_config({"gemini_env_file_path": str(resolved)})
    return resolved


def clear_configured_gemini_env_file() -> None:
    path = gemini_auth_config_path()
    if path.exists():
        path.unlink()


def _inspect_keyring() -> tuple[Any | None, str | None]:
    try:
        import keyring
    except Exception:
        return None, None

    try:
        backend = keyring.get_keyring()
        backend_name = f"{backend.__class__.__module__}.{backend.__class__.__name__}"
        priority = getattr(backend, "priority", 1)
        if priority is None or priority <= 0:
            return None, backend_name
        return keyring, backend_name
    except Exception:
        return None, None


def _require_keyring() -> tuple[Any, str]:
    keyring_module, backend_name = _inspect_keyring()
    if keyring_module is None:
        raise RuntimeError(
            "A usable keyring backend is not available. "
            "Configure Gemini with `istots auth gemini env-file set PATH` instead."
        )
    return keyring_module, str(backend_name)


def set_gemini_api_key(api_key: str) -> str:
    normalized = api_key.strip()
    if not normalized:
        raise RuntimeError("Gemini API key must not be empty.")
    keyring_module, backend_name = _require_keyring()
    keyring_module.set_password(GEMINI_KEYRING_SERVICE, GEMINI_KEYRING_USERNAME, normalized)
    return backend_name


def delete_gemini_api_key() -> str | None:
    keyring_module, backend_name = _require_keyring()
    try:
        keyring_module.delete_password(GEMINI_KEYRING_SERVICE, GEMINI_KEYRING_USERNAME)
    except Exception:
        pass
    return backend_name


def get_stored_gemini_api_key() -> tuple[str | None, str | None]:
    keyring_module, backend_name = _inspect_keyring()
    if keyring_module is None:
        return None, backend_name
    try:
        return keyring_module.get_password(GEMINI_KEYRING_SERVICE, GEMINI_KEYRING_USERNAME), backend_name
    except Exception:
        return None, backend_name


def _candidate_api_key_names(api_key_env: str) -> tuple[str, ...]:
    names = [api_key_env]
    if api_key_env == "GEMINI_API_KEY":
        names.append("GOOGLE_API_KEY")
    return tuple(dict.fromkeys(names))


def _parse_env_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    if not path.exists():
        return assignments
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if (
            len(normalized_value) >= 2
            and normalized_value[0] == normalized_value[-1]
            and normalized_value[0] in {"'", '"'}
        ):
            normalized_value = normalized_value[1:-1]
        assignments[normalized_key] = normalized_value
    return assignments


def resolve_gemini_api_key(api_key_env: str = "GEMINI_API_KEY") -> tuple[str | None, str | None]:
    stored_key, _ = get_stored_gemini_api_key()
    if stored_key:
        return stored_key, "keyring"

    env_file = get_configured_gemini_env_file()
    if env_file is not None:
        env_values = _parse_env_assignments(env_file)
        for key_name in _candidate_api_key_names(api_key_env):
            candidate = env_values.get(key_name)
            if candidate:
                return candidate, f"env-file:{env_file}"

    for key_name in _candidate_api_key_names(api_key_env):
        candidate = os.environ.get(key_name)
        if candidate:
            return candidate, f"environment:{key_name}"
    return None, None


def get_gemini_auth_status(api_key_env: str = "GEMINI_API_KEY") -> GeminiAuthStatus:
    stored_key, backend_name = get_stored_gemini_api_key()
    env_file = get_configured_gemini_env_file()
    env_file_values = _parse_env_assignments(env_file) if env_file is not None else {}
    env_file_contains_key = any(
        bool(env_file_values.get(key_name))
        for key_name in _candidate_api_key_names(api_key_env)
    )
    process_env_name: str | None = None
    process_env_configured = False
    for key_name in _candidate_api_key_names(api_key_env):
        if os.environ.get(key_name):
            process_env_name = key_name
            process_env_configured = True
            break

    _, effective_source = resolve_gemini_api_key(api_key_env)
    return GeminiAuthStatus(
        keyring_backend=backend_name,
        keyring_available=backend_name is not None,
        keyring_configured=bool(stored_key),
        env_file_path=env_file,
        env_file_exists=bool(env_file and env_file.exists()),
        env_file_configured=env_file is not None,
        env_file_contains_key=env_file_contains_key,
        process_env_name=process_env_name,
        process_env_configured=process_env_configured,
        effective_source=effective_source,
    )
