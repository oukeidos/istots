from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from istots.llama_runtime import (
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
    models_dir: Path | None = None
    role: str = "ocr"
    profile: str = "auto"
    binary_path: Path | None = None
    host: str = "127.0.0.1"
    port: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
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

        overrides = LlamaServerOverrides(
            profile=normalize_llama_server_profile(self.profile),
            device=self.device,
            threads=self.threads,
            threads_batch=self.threads_batch,
            port=self.port,
            gpu_layers=self.gpu_layers,
            no_mmproj_offload=self.no_mmproj_offload,
        )
        self._launch_spec = build_llama_server_launch_spec(
            role=normalize_llama_server_role(self.role),
            binary_path=resolved_binary,
            models_dir=self.models_dir,
            host=self.host,
            overrides=overrides,
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
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images: Sequence[Image.Image]) -> list[str]:
        if not images:
            return []
        texts = [
            request_llama_server_ocr(
                self._launch_spec,
                image,
                max_new_tokens=self.max_new_tokens,
                prompt_text=self.prompt_text,
            )
            for image in images
        ]
        return [normalize_ocr_text(text) for text in texts]

    def clear_device_cache(self) -> None:
        return None

    def close(self) -> None:
        process = getattr(self, "_process", None)
        self._process = None
        if process is not None:
            stop_llama_server(process)
