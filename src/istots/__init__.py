"""istots package."""

from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
import tomllib


def _read_version() -> str:
    try:
        return _dist_version("istots")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.exists():
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            return data["project"]["version"]
        return "0+unknown"


__version__ = _read_version()
