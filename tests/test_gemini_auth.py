from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from istots import gemini_auth


class _FakeKeyringBackend:
    priority = 1


def _install_fake_keyring(monkeypatch, *, stored_value: str | None = None):
    state = {"value": stored_value}

    fake_keyring = SimpleNamespace(
        get_keyring=lambda: _FakeKeyringBackend(),
        get_password=lambda service, username: state["value"],
        set_password=lambda service, username, value: state.__setitem__("value", value),
        delete_password=lambda service, username: state.__setitem__("value", None),
    )
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)
    return state


def test_configured_gemini_env_file_round_trip(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "auth.json"
    env_path = tmp_path / ".env"
    monkeypatch.setenv("ISTOTS_AUTH_CONFIG_PATH", str(config_path))

    resolved = gemini_auth.set_configured_gemini_env_file(env_path)

    assert resolved == env_path.resolve()
    assert gemini_auth.get_configured_gemini_env_file() == env_path.resolve()

    gemini_auth.clear_configured_gemini_env_file()

    assert gemini_auth.get_configured_gemini_env_file() is None


def test_set_gemini_api_key_uses_keyring(monkeypatch) -> None:
    state = _install_fake_keyring(monkeypatch)

    backend_name = gemini_auth.set_gemini_api_key("  secret-key  ")
    resolved_key, resolved_backend = gemini_auth.get_stored_gemini_api_key()

    assert backend_name.endswith("_FakeKeyringBackend")
    assert resolved_backend == backend_name
    assert resolved_key == "secret-key"
    assert state["value"] == "secret-key"


def test_resolve_gemini_api_key_prefers_keyring_over_env_file_and_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _install_fake_keyring(monkeypatch, stored_value="keyring-secret")
    config_path = tmp_path / "auth.json"
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=env-file-secret\n", encoding="utf-8")
    monkeypatch.setenv("ISTOTS_AUTH_CONFIG_PATH", str(config_path))
    gemini_auth.set_configured_gemini_env_file(env_path)
    monkeypatch.setenv("GEMINI_API_KEY", "environment-secret")

    api_key, source = gemini_auth.resolve_gemini_api_key()

    assert api_key == "keyring-secret"
    assert source == "keyring"


def test_resolve_gemini_api_key_uses_env_file_before_environment(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "auth.json"
    env_path = tmp_path / ".env"
    env_path.write_text("export GOOGLE_API_KEY='env-file-secret'\n", encoding="utf-8")
    monkeypatch.setenv("ISTOTS_AUTH_CONFIG_PATH", str(config_path))
    gemini_auth.set_configured_gemini_env_file(env_path)
    monkeypatch.setenv("GEMINI_API_KEY", "environment-secret")

    api_key, source = gemini_auth.resolve_gemini_api_key()

    assert api_key == "env-file-secret"
    assert source == f"env-file:{env_path.resolve()}"


def test_get_gemini_auth_status_reports_effective_source(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "auth.json"
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=env-file-secret\n", encoding="utf-8")
    monkeypatch.setenv("ISTOTS_AUTH_CONFIG_PATH", str(config_path))
    gemini_auth.set_configured_gemini_env_file(env_path)

    status = gemini_auth.get_gemini_auth_status()

    assert status.keyring_configured is False
    assert status.env_file_configured is True
    assert status.env_file_contains_key is True
    assert status.process_env_configured is False
    assert status.effective_source == f"env-file:{env_path.resolve()}"


def test_set_gemini_api_key_raises_without_usable_keyring(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(get_keyring=lambda: SimpleNamespace(priority=0)),
    )

    with pytest.raises(RuntimeError, match="usable keyring backend is not available"):
        gemini_auth.set_gemini_api_key("secret-key")
