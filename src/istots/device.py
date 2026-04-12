from __future__ import annotations

from typing import Any


def resolve_device(preferred: str) -> str:
    requested = preferred.lower()
    if requested == "cuda":
        requested = "gpu"
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported device: {preferred}")

    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if not has_gpu():
            raise RuntimeError("GPU device requested but no compatible GPU is available.")
        return "gpu"

    return "gpu" if has_gpu() else "cpu"


def has_gpu() -> bool:
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


def pick_torch_dtype(device: str) -> Any:
    import torch

    if to_torch_device(device) == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
