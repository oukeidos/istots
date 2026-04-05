from __future__ import annotations

from typing import Any


def resolve_device(preferred: str) -> str:
    requested = preferred.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {preferred}")

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not has_cuda():
            raise RuntimeError("CUDA device requested but no CUDA GPU is available.")
        return "cuda"

    return "cuda" if has_cuda() else "cpu"


def has_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pick_torch_dtype(device: str) -> Any:
    import torch

    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
