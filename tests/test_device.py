from __future__ import annotations

import pytest

from istots import device


def test_resolve_device_prefers_gpu_when_available(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_gpu", lambda: True)
    assert device.resolve_device("auto") == "gpu"


def test_resolve_device_falls_back_to_cpu_when_gpu_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_gpu", lambda: False)
    assert device.resolve_device("auto") == "cpu"


def test_resolve_device_accepts_cuda_as_legacy_alias(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_gpu", lambda: True)
    assert device.resolve_device("cuda") == "gpu"


def test_resolve_device_rejects_gpu_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_gpu", lambda: False)
    with pytest.raises(RuntimeError, match="no compatible GPU"):
        device.resolve_device("gpu")


def test_to_torch_device_maps_generic_gpu_to_cuda() -> None:
    assert device.to_torch_device("gpu") == "cuda"
    assert device.to_torch_device("cpu") == "cpu"
