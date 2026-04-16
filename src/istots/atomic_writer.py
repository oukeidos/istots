from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO


def _fsync_file(handle: TextIO) -> None:
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def atomic_write_text_file(
    path: Path,
    write_text: Callable[[TextIO], None],
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    replaced = False
    try:
        with temp_path.open("w", encoding=encoding, newline=newline) as handle:
            write_text(handle)
            handle.flush()
            _fsync_file(handle)
        temp_path.replace(path)
        replaced = True
    finally:
        if not replaced:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    atomic_write_text_file(
        path,
        lambda handle: handle.write(text),
        encoding=encoding,
        newline=newline,
    )


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = None,
    trailing_newline: bool = True,
) -> None:
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    if trailing_newline:
        text += "\n"
    atomic_write_text(path, text, encoding="utf-8")


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[Any],
    *,
    ensure_ascii: bool = False,
) -> None:
    def write_rows(handle: TextIO) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=ensure_ascii))
            handle.write("\n")

    atomic_write_text_file(path, write_rows, encoding="utf-8")
