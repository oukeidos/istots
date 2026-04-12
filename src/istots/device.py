from __future__ import annotations

from typing import Any


def resolve_hf_device(preferred: str) -> str:
    requested = preferred.lower()
    if requested == "cuda":
        requested = "gpu"
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported device: {preferred}")

    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if not has_hf_gpu():
            raise RuntimeError("GPU device requested but no compatible GPU is available.")
        return "gpu"

    return "gpu" if has_hf_gpu() else "cpu"


def has_hf_gpu() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def to_torch_device(device: str) -> str:
    normalized = device.lower()
    if normalized == "cuda":
        normalized = "gpu"
    if normalized == "gpu":
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported torch device mapping: {device}")


def supports_cpu_bfloat16() -> bool:
    import torch

    mkldnn_backend = getattr(getattr(torch, "backends", None), "mkldnn", None)
    if mkldnn_backend is not None and hasattr(mkldnn_backend, "is_available"):
        try:
            if not bool(mkldnn_backend.is_available()):
                return False
        except Exception:
            return False

    autocast_mode = getattr(getattr(torch, "amp", None), "autocast_mode", None)
    if autocast_mode is not None and hasattr(autocast_mode, "is_autocast_available"):
        try:
            return bool(autocast_mode.is_autocast_available("cpu"))
        except Exception:
            return False

    return False


def pick_torch_dtype(device: str, preferred: str = "auto") -> Any:
    import torch

    requested = preferred.lower()
    if requested not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported torch dtype preference: {preferred}")

    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16

    if to_torch_device(device) == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if supports_cpu_bfloat16():
        return torch.bfloat16
    return torch.float32
