from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from istots import device


def test_resolve_hf_device_prefers_gpu_when_available(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_hf_gpu", lambda: True)
    assert device.resolve_hf_device("auto") == "gpu"


def test_resolve_hf_device_falls_back_to_cpu_when_gpu_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_hf_gpu", lambda: False)
    assert device.resolve_hf_device("auto") == "cpu"


def test_resolve_hf_device_accepts_cuda_as_legacy_alias(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_hf_gpu", lambda: True)
    assert device.resolve_hf_device("cuda") == "gpu"


def test_resolve_hf_device_rejects_gpu_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(device, "has_hf_gpu", lambda: False)
    with pytest.raises(RuntimeError, match="no compatible GPU"):
        device.resolve_hf_device("gpu")


def test_to_torch_device_maps_generic_gpu_to_cuda() -> None:
    assert device.to_torch_device("gpu") == "cuda"
    assert device.to_torch_device("cpu") == "cpu"


def test_pick_torch_dtype_prefers_bfloat16_for_supported_cpu(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="f16",
        float32="f32",
        cuda=SimpleNamespace(is_bf16_supported=lambda: True),
        backends=SimpleNamespace(mkldnn=SimpleNamespace(is_available=lambda: True)),
        amp=SimpleNamespace(
            autocast_mode=SimpleNamespace(is_autocast_available=lambda target: target == "cpu")
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert device.supports_cpu_bfloat16() is True
    assert device.pick_torch_dtype("cpu") == "bf16"


def test_pick_torch_dtype_uses_explicit_override(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="f16",
        float32="f32",
        cuda=SimpleNamespace(is_bf16_supported=lambda: False),
        backends=SimpleNamespace(mkldnn=SimpleNamespace(is_available=lambda: False)),
        amp=SimpleNamespace(
            autocast_mode=SimpleNamespace(is_autocast_available=lambda target: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert device.pick_torch_dtype("cpu", "float32") == "f32"
    assert device.pick_torch_dtype("gpu", "float16") == "f16"
