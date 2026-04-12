from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from istots.device import pick_torch_dtype, to_torch_device

OCR_PROMPT = "OCR:"
ROPE_WARNING_LOGGER = "transformers.modeling_rope_utils"


@dataclass
class HFPaddleOCRVLBackend:
    model_id: str
    device: str
    max_new_tokens: int = 256
    local_files_only: bool = True

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "transformers/torch are required for OCR inference. "
                "Install project dependencies first."
            ) from exc

        # PaddleOCR-VL models can emit a known RoPE validation warning
        # about `mrope_section`; it is non-fatal and can be safely muted.
        logging.getLogger(ROPE_WARNING_LOGGER).setLevel(logging.ERROR)

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        )
        self._torch_device = to_torch_device(self.device)
        if hasattr(self._processor, "tokenizer") and hasattr(self._processor.tokenizer, "padding_side"):
            self._processor.tokenizer.padding_side = "left"
        elif hasattr(self._processor, "padding_side"):
            self._processor.padding_side = "left"
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=pick_torch_dtype(self.device),
            local_files_only=self.local_files_only,
        )
        self._model.to(self._torch_device)
        self._model.eval()

    def recognize(self, image: Image.Image) -> str:
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images: Sequence[Image.Image]) -> list[str]:
        if not images:
            return []

        prompt = OCR_PROMPT

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image in images
        ]

        inputs = self._processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        model_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                model_inputs[key] = value.to(self._model.device)
            else:
                model_inputs[key] = value

        with self._torch.inference_mode():
            output_tokens = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = model_inputs["input_ids"].shape[-1]
        generated_tokens = output_tokens[:, prompt_len:]
        texts = self._processor.batch_decode(generated_tokens, skip_special_tokens=True)
        return [normalize_ocr_text(text) for text in texts]

    def clear_device_cache(self) -> None:
        if self._torch_device != "cuda" or not hasattr(self, "_torch") or not hasattr(self._torch, "cuda"):
            return
        if hasattr(self._torch.cuda, "empty_cache"):
            self._torch.cuda.empty_cache()
        if hasattr(self._torch.cuda, "ipc_collect"):
            self._torch.cuda.ipc_collect()

    def close(self) -> None:
        import gc

        model = getattr(self, "_model", None)
        processor = getattr(self, "_processor", None)

        self._model = None
        self._processor = None

        if model is not None:
            del model
        if processor is not None:
            del processor

        self.clear_device_cache()
        gc.collect()


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<\|[^>]+\|>", "", text)
    text = text.replace("```text", "").replace("```", "")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()
