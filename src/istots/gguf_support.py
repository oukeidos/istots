from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import urllib.request
from pathlib import Path

GGUF_PY_COMMIT = "94ca829b6001019622c0f67fcd48e9ec6bd7dce8"
GGUF_PY_VERSION = "0.18.0"
GGUF_PY_DEFAULT_BASE_URL = (
    f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{GGUF_PY_COMMIT}/gguf-py"
)

GGUF_PY_FILES: dict[str, str] = {
    "LICENSE": "ef78c7e6659e34c798194f0d344e807c722de70cbca1359ef54772661e11ca38",
    "gguf/__init__.py": "3ccfc0104cd7ea88c6028743b7bf3f2c89b5f474425de03a217a6072320d7c2f",
    "gguf/constants.py": "b7c87aa440ab4324733e11ffd6224b20cea738104b5748e1d602eacc11faaa66",
    "gguf/gguf.py": "f0c0eeedad0911784b52ffed8e162a0eb5ae6d535ce35705bc16196e46597a72",
    "gguf/gguf_reader.py": "91cb9e81f1a67f5856f012cf3f526ad9ffdb6b887d2aa66e2593ee6f63764aad",
    "gguf/gguf_writer.py": "890478c9f62bc35461f279a94ef3eae29dbfdc6f8b9ac5d192693980b18eea0e",
    "gguf/lazy.py": "ec35541320443e1bdc293e39c699fa5d1adb7fb594f3f87af093b4c756a4ebe2",
    "gguf/metadata.py": "c871748f49f7fb8e67aa9a18867adf8a8ba862f0f1f4a89a0a0e285daa52a312",
    "gguf/py.typed": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "gguf/quants.py": "db403c3b2292d3f2c5cfef4109d4b5745f437b5599c7afc94d4b97feca7e9247",
    "gguf/tensor_mapping.py": "21e7f8db3ec451ac5704c468f85edbfaec8d0eeaf7425d5073ca3f45e8aa7694",
    "gguf/utility.py": "da920e2c62166ec9f30e407e9fb35ec29fda671310ffbf0f8ce856b62f1c70d7",
    "gguf/vocab.py": "9cbc1afea31a12858315bec648f148194d5eaa9c9eaf16e8ad6354b531f17ba3",
}

MANIFEST_NAME = "snapshot_manifest.json"


def default_support_dir() -> Path:
    configured = os.environ.get("ISTOTS_SUPPORT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "istots" / "support").resolve()


def gguf_py_snapshot_dir(support_dir: Path | None = None) -> Path:
    root = (support_dir or default_support_dir()).expanduser().resolve()
    return root / "gguf-py" / GGUF_PY_COMMIT


def ensure_known_good_gguf_py(
    *,
    support_dir: Path | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> Path:
    snapshot_dir = gguf_py_snapshot_dir(support_dir)
    source_root = (base_url or os.environ.get("ISTOTS_GGUF_PY_BASE_URL") or GGUF_PY_DEFAULT_BASE_URL).strip()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for relative_path, expected_sha256 in GGUF_PY_FILES.items():
        target = snapshot_dir / relative_path
        if not force and _is_file_hash_match(target, expected_sha256):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _fetch_source_bytes(source_root, relative_path)
        actual_sha256 = _sha256_bytes(data)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "known-good gguf-py download failed hash check for "
                f"{relative_path}: expected {expected_sha256}, got {actual_sha256}"
            )
        temp_target = target.with_name(f".{target.name}.tmp")
        temp_target.write_bytes(data)
        temp_target.replace(target)

    manifest_path = snapshot_dir / MANIFEST_NAME
    manifest_payload = {
        "commit": GGUF_PY_COMMIT,
        "version": GGUF_PY_VERSION,
        "source_root": source_root,
        "files": GGUF_PY_FILES,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot_dir


def get_installed_gguf():
    return _import_gguf_if_available()


def load_known_good_gguf(
    *,
    source_mode: str = "auto",
    support_dir: Path | None = None,
    base_url: str | None = None,
) -> object:
    if source_mode not in {"auto-download", "installed", "auto"}:
        raise RuntimeError(f"unsupported gguf source mode: {source_mode}")

    if source_mode in {"installed", "auto"}:
        module = get_installed_gguf()
        if module is not None:
            return module
        if source_mode == "installed":
            raise RuntimeError(
                "no installed gguf package is available. "
                "Install gguf or use --gguf-source-mode auto-download."
            )

    snapshot_dir = ensure_known_good_gguf_py(
        support_dir=support_dir,
        base_url=base_url,
    )
    return import_known_good_gguf(snapshot_dir)


def import_known_good_gguf(snapshot_dir: Path):
    snapshot_dir = snapshot_dir.expanduser().resolve()
    for name in list(sys.modules):
        if name == "gguf" or name.startswith("gguf."):
            sys.modules.pop(name, None)

    sys.path.insert(0, str(snapshot_dir))
    try:
        module = importlib.import_module("gguf")
    finally:
        try:
            sys.path.remove(str(snapshot_dir))
        except ValueError:
            pass

    module_path = Path(getattr(module, "__file__", "")).resolve()
    expected_root = (snapshot_dir / "gguf").resolve()
    if expected_root not in module_path.parents and module_path != expected_root / "__init__.py":
        raise RuntimeError(
            "imported gguf module does not match the pinned local snapshot: "
            f"{module_path}"
        )
    return module


def _import_gguf_if_available():
    try:
        return importlib.import_module("gguf")
    except Exception:
        return None


def _fetch_source_bytes(source_root: str, relative_path: str) -> bytes:
    source_root_path = Path(source_root).expanduser()
    if source_root_path.exists():
        return (source_root_path / relative_path).read_bytes()

    root = source_root.rstrip("/")
    url = f"{root}/{relative_path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_file_hash_match(path: Path, expected_sha256: str) -> bool:
    return path.exists() and path.is_file() and _sha256_path(path) == expected_sha256
