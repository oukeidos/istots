from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from istots.app.auth import AuthRequest, execute_auth_request
from istots.gemini_auth import GeminiAuthStatus


def test_execute_auth_request_status_returns_structured_status(monkeypatch) -> None:
    status = GeminiAuthStatus(
        keyring_backend="test.backend",
        keyring_available=True,
        keyring_configured=True,
        env_file_path=Path("/tmp/.env"),
        env_file_exists=True,
        env_file_configured=True,
        env_file_contains_key=True,
        process_env_name=None,
        process_env_configured=False,
        effective_source="keyring",
    )

    monkeypatch.setattr(
        "istots.app.auth.gemini_auth_module.get_gemini_auth_status",
        lambda: status,
    )

    result = execute_auth_request(AuthRequest(provider="gemini", action="status"))

    assert result.action == "status"
    assert result.status == status


def test_execute_auth_request_env_file_set_returns_resolved_path(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"

    monkeypatch.setattr(
        "istots.app.auth.gemini_auth_module.set_configured_gemini_env_file",
        lambda path: path.resolve(),
    )

    result = execute_auth_request(
        AuthRequest(
            provider="gemini",
            action="env-file",
            env_file_action="set",
            path=env_path,
        )
    )

    assert result.env_file_action == "set"
    assert result.resolved_path == env_path.resolve()


def test_execute_auth_request_set_uses_supplied_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "istots.app.auth.gemini_auth_module.set_gemini_api_key",
        lambda api_key: "test.backend" if api_key == "secret-key" else "wrong",
    )

    result = execute_auth_request(
        AuthRequest(
            provider="gemini",
            action="set",
            api_key="secret-key",
        )
    )

    assert result.action == "set"
    assert result.backend_name == "test.backend"
