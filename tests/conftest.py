from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_prepared_input_subprocess_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISTOTS_PREPARE_OCR_INPUTS_IN_SUBPROCESS", "0")
