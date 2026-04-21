from __future__ import annotations

import sys
from io import StringIO

from istots.runtime_stdio import ensure_standard_streams


def test_ensure_standard_streams_restores_original_stderr(monkeypatch) -> None:
    stderr = StringIO()

    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "__stderr__", stderr, raising=False)

    ensure_standard_streams()

    assert sys.stderr is stderr


def test_ensure_standard_streams_supplies_devnull_when_both_stderr_handles_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "__stderr__", None, raising=False)

    ensure_standard_streams()

    assert sys.stderr is not None
    assert callable(getattr(sys.stderr, "write", None))
    sys.stderr.write("safe\n")


def test_ensure_standard_streams_supplies_devnull_when_stdout_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "__stdout__", None, raising=False)

    ensure_standard_streams()

    assert sys.stdout is not None
    assert callable(getattr(sys.stdout, "write", None))
    sys.stdout.write("safe\n")
