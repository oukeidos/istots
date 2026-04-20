from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from istots import gemini_auth as gemini_auth_module
from istots.gemini_auth import GeminiAuthStatus


class AuthArgumentError(ValueError):
    pass


class AuthExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthRequest:
    provider: str
    action: str
    env_file_action: str | None = None
    path: Path | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class AuthResult:
    provider: str
    action: str
    env_file_action: str | None = None
    backend_name: str | None = None
    status: GeminiAuthStatus | None = None
    resolved_path: Path | None = None


def execute_auth_request(request: AuthRequest) -> AuthResult:
    if request.provider != "gemini":
        raise AuthArgumentError(f"unsupported auth provider: {request.provider}")

    try:
        if request.action == "set":
            if request.api_key is None:
                raise AuthArgumentError("Gemini API key is required for auth set.")
            backend_name = gemini_auth_module.set_gemini_api_key(request.api_key)
            return AuthResult(provider="gemini", action="set", backend_name=backend_name)
        if request.action == "delete":
            backend_name = gemini_auth_module.delete_gemini_api_key()
            return AuthResult(provider="gemini", action="delete", backend_name=backend_name)
        if request.action == "status":
            status = gemini_auth_module.get_gemini_auth_status()
            return AuthResult(provider="gemini", action="status", status=status)
        if request.action == "env-file":
            if request.env_file_action == "set":
                if request.path is None:
                    raise AuthArgumentError("Path is required for auth env-file set.")
                resolved_path = gemini_auth_module.set_configured_gemini_env_file(request.path)
                return AuthResult(
                    provider="gemini",
                    action="env-file",
                    env_file_action="set",
                    resolved_path=resolved_path,
                )
            if request.env_file_action == "clear":
                gemini_auth_module.clear_configured_gemini_env_file()
                return AuthResult(provider="gemini", action="env-file", env_file_action="clear")
    except AuthArgumentError:
        raise
    except Exception as exc:
        raise AuthExecutionError(str(exc)) from exc

    raise AuthArgumentError("unsupported auth action")
