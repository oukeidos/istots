from __future__ import annotations

import atexit
import os
import sys
from collections.abc import Callable
from io import TextIOBase

_DEVNULL_STREAMS: dict[str, TextIOBase] = {}
_CLEANUP_REGISTERED = False


def ensure_standard_streams() -> None:
    _ensure_text_stream("stdin", mode="r", predicate=lambda stream: callable(getattr(stream, "read", None)))
    _ensure_text_stream("stdout", mode="a", predicate=lambda stream: callable(getattr(stream, "write", None)))
    _ensure_text_stream("stderr", mode="a", predicate=lambda stream: callable(getattr(stream, "write", None)))


def _ensure_text_stream(
    stream_name: str,
    *,
    mode: str,
    predicate: Callable[[object], bool],
) -> None:
    current_stream = getattr(sys, stream_name, None)
    if predicate(current_stream):
        if getattr(sys, f"__{stream_name}__", None) is None:
            setattr(sys, f"__{stream_name}__", current_stream)
        return

    original_stream = getattr(sys, f"__{stream_name}__", None)
    if predicate(original_stream):
        setattr(sys, stream_name, original_stream)
        return

    fallback_stream = _get_devnull_text_stream(stream_name, mode=mode)
    setattr(sys, stream_name, fallback_stream)
    if getattr(sys, f"__{stream_name}__", None) is None:
        setattr(sys, f"__{stream_name}__", fallback_stream)


def _get_devnull_text_stream(stream_name: str, *, mode: str) -> TextIOBase:
    stream = _DEVNULL_STREAMS.get(stream_name)
    if stream is not None and not stream.closed:
        return stream

    stream = open(os.devnull, mode, encoding="utf-8", errors="replace")
    _DEVNULL_STREAMS[stream_name] = stream
    _register_cleanup()
    return stream


def _register_cleanup() -> None:
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    atexit.register(_close_devnull_streams)
    _CLEANUP_REGISTERED = True


def _close_devnull_streams() -> None:
    for stream in list(_DEVNULL_STREAMS.values()):
        try:
            stream.close()
        except Exception:
            pass
    _DEVNULL_STREAMS.clear()
