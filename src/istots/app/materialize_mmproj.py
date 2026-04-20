from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from istots import llama_mmproj as llama_mmproj_module


class MaterializeMmprojExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializeMmprojRequest:
    base_mmproj: Path
    output: Path | None = None
    min_pixels: int = 32768
    support_dir: Path | None = None
    gguf_py_base_url: str | None = None
    gguf_source_mode: str = "auto"
    force: bool = False


@dataclass(frozen=True)
class MaterializeMmprojResult:
    output: Path
    applied_value: int


def execute_materialize_mmproj_request(
    request: MaterializeMmprojRequest,
) -> MaterializeMmprojResult:
    try:
        output = llama_mmproj_module.materialize_mmproj(
            base_mmproj=request.base_mmproj,
            output_path=request.output,
            min_pixels=request.min_pixels,
            support_dir=request.support_dir,
            gguf_py_base_url=request.gguf_py_base_url,
            gguf_source_mode=request.gguf_source_mode,
            force=request.force,
        )
        applied_value = llama_mmproj_module.read_mmproj_min_pixels(
            output,
            support_dir=request.support_dir,
            gguf_py_base_url=request.gguf_py_base_url,
            gguf_source_mode=request.gguf_source_mode,
        )
    except Exception as exc:
        raise MaterializeMmprojExecutionError(str(exc)) from exc

    return MaterializeMmprojResult(output=output, applied_value=applied_value)
