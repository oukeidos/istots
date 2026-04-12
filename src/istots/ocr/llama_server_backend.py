from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from PIL import Image

from istots.llama_runtime import (
    DEFAULT_ROLE_PORTS,
    LlamaServerLaunchSpec,
    LlamaServerOverrides,
    build_llama_server_launch_spec,
    detect_llama_server_path,
    normalize_llama_server_profile,
    normalize_llama_server_role,
    request_llama_server_ocr,
    start_llama_server,
    stop_llama_server,
)

from .hf_backend import normalize_ocr_text


@dataclass
class LlamaServerOCRBackend:
    device: str
    max_new_tokens: int = 256
    model_path: Path | None = None
    mmproj_path: Path | None = None
    models_dir: Path | None = None
    role: str = "ocr"
    profile: str = "auto"
    binary_path: Path | None = None
    host: str = "127.0.0.1"
    port: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    ctx_size: int | None = None
    n_predict: int | None = None
    reasoning: str | None = None
    reasoning_budget: int | None = None
    gpu_layers: int | None = None
    no_mmproj_offload: bool | None = None
    startup_timeout_sec: float = 120.0
    prompt_text: str = "OCR:"

    def __post_init__(self) -> None:
        resolved_binary = detect_llama_server_path(self.binary_path)
        if resolved_binary is None:
            raise RuntimeError(
                "llama-server binary not found. Set ISTOTS_LLAMA_SERVER_PATH or pass --llama-server-path."
            )

        normalized_role = normalize_llama_server_role(self.role)
        normalized_profile = normalize_llama_server_profile(self.profile)

        if (self.model_path is None) != (self.mmproj_path is None):
            raise RuntimeError(
                "explicit llama-server model_path and mmproj_path must be provided together"
            )

        if self.model_path is not None and self.mmproj_path is not None:
            self._launch_spec = LlamaServerLaunchSpec(
                role=normalized_role,
                profile=normalized_profile,
                binary_path=resolved_binary,
                model_path=self.model_path.expanduser().resolve(),
                mmproj_path=self.mmproj_path.expanduser().resolve(),
                host=self.host,
                port=self.port or DEFAULT_ROLE_PORTS[normalized_role],
                device=self.device,
                threads=self.threads,
                threads_batch=self.threads_batch,
                ctx_size=self.ctx_size,
                n_predict=self.n_predict,
                reasoning=self.reasoning,
                reasoning_budget=self.reasoning_budget,
                gpu_layers=self.gpu_layers,
                no_mmproj_offload=bool(self.no_mmproj_offload),
                prompt_text=self.prompt_text,
            )
        else:
            overrides = LlamaServerOverrides(
                profile=normalized_profile,
                device=self.device,
                threads=self.threads,
                threads_batch=self.threads_batch,
                port=self.port,
                gpu_layers=self.gpu_layers,
                no_mmproj_offload=self.no_mmproj_offload,
            )
            self._launch_spec = build_llama_server_launch_spec(
                role=normalized_role,
                binary_path=resolved_binary,
                models_dir=self.models_dir,
                host=self.host,
                overrides=overrides,
            )
            if (
                self.prompt_text != "OCR:"
                or self.ctx_size is not None
                or self.n_predict is not None
                or self.reasoning is not None
                or self.reasoning_budget is not None
            ):
                self._launch_spec = replace(
                    self._launch_spec,
                    prompt_text=self.prompt_text,
                    ctx_size=self.ctx_size,
                    n_predict=self.n_predict,
                    reasoning=self.reasoning,
                    reasoning_budget=self.reasoning_budget,
                )
        missing_paths = [
            path for path in (self._launch_spec.model_path, self._launch_spec.mmproj_path) if not path.exists()
        ]
        if missing_paths:
            joined = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(f"required llama-server runtime assets are missing: {joined}")

        self._process = start_llama_server(
            self._launch_spec,
            startup_timeout_sec=self.startup_timeout_sec,
        )

    def recognize(self, image: Image.Image) -> str:
        return normalize_ocr_text(
            request_llama_server_ocr(
                self._launch_spec,
                image,
                max_new_tokens=self.max_new_tokens,
                prompt_text=self.prompt_text,
            )
        )

    def recognize_batch(self, images: Sequence[Image.Image]) -> list[str]:
        if not images:
            return []
        return [self.recognize(image) for image in images]

    def clear_device_cache(self) -> None:
        return None

    def close(self) -> None:
        process = getattr(self, "_process", None)
        self._process = None
        if process is not None:
            stop_llama_server(process)
