# Third-Party Notices

This document records the third-party packages, model sources, runtimes, and
remote services used by the current `istots` repository state.

## Inventory Basis

This notice file was prepared from:

- `pyproject.toml`
- `uv.lock` (revision `3`)
- source inspection of:
  - `src/istots/model_store.py`
  - `src/istots/gguf_support.py`
  - `src/istots/llama_runtime.py`
  - `src/istots/corrector.py`
  - `tools/image_compare_gui/compare_images.py`

Important scope notes:

- The exact installed package set varies by Python version, operating system,
  architecture, and whether the optional `hf` extra is installed.
- `llama-server` is not vendored by this repository and is not installed by the
  Python package. `istots` discovers a user- or host-provided binary at run
  time.
- Model files are downloaded into a local cache by `istots setup` and are not
  vendored in this repository.
- Gemini correction uses the remote Google Generative Language API over HTTP.
  No Google Gemini Python SDK is bundled here.
- The optional local image comparison GUI in `tools/image_compare_gui/` uses
  Python's standard-library `tkinter` module and therefore depends on the host
  Python runtime's Tcl/Tk packaging.

## First-Party License

- `istots`: MIT. See `LICENSE`.

## Host-Provided Runtimes and Remote Services

| Component | Source | License / Terms | Notes |
| --- | --- | --- | --- |
| `llama-server` runtime binary | User- or host-provided binary, typically from the `llama.cpp` project | Upstream `llama.cpp` terms and bundled notices apply to the binary actually installed on the host | `istots` probes `ISTOTS_LLAMA_SERVER_PATH`, `PATH`, and a fallback local path. The binary is required for the primary OCR path, detector path, and local Qwen corrector path. |
| `tkinter` / Tcl-Tk runtime | Host Python distribution | Tcl/Tk license terms as bundled with the host Python distribution | Used only by the optional local image-comparison GUI in `tools/image_compare_gui/compare_images.py`. |
| Google Generative Language API (`Gemini`) | Remote Google service | Google API and model terms apply | Used only when `--corrector gemini` is selected. No model weights or SDK are redistributed by this repository. |

## External Model and Artifact Sources

| Artifact | Source | License / Terms | Notes |
| --- | --- | --- | --- |
| HF fallback OCR model | <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5> | `apache-2.0` according to the upstream model card | Used only by the optional `--engine hf` path. Downloaded by `istots setup`. |
| Primary OCR GGUF model and base mmproj | <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-GGUF> | Review the upstream repository license at download time | `istots` downloads `PaddleOCR-VL-1.5.gguf` and `PaddleOCR-VL-1.5-mmproj.gguf` for the retained primary OCR path. |
| Derived OCR mmproj | Locally materialized from the official Paddle GGUF base mmproj | Inherits the applicable terms of the upstream source artifact | `istots` creates a local derived `min_pixels=32768` mmproj from the official base mmproj; this is not downloaded as a separate upstream artifact. |
| Optional local Qwen corrector model and mmproj | <https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF> | Review the upstream repository license at download time | When `istots setup --with-qwen-corrector` is used, the default files are `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` and `mmproj-BF16.gguf`. |
| Optional pinned `gguf-py` snapshot fallback | <https://github.com/ggml-org/llama.cpp/tree/94ca829b6001019622c0f67fcd48e9ec6bd7dce8/gguf-py> | MIT | The project primarily uses the installed `gguf` package and can also auto-download this pinned fallback snapshot into a local support cache when `--gguf-source-mode auto-download` is selected or no installed package is available in `auto` mode. |

## Direct Python Dependencies

These packages are declared directly in `pyproject.toml`.

### Core Runtime

| Package | Version | License | Notes |
| --- | --- | --- | --- |
| `gguf` | `0.18.0` | MIT | Declared as the adopted PyPI release package. Used for GGUF mmproj inspection and materialization. |
| `huggingface-hub` | `1.4.1` | Apache | Used by `istots setup` to download HF and GGUF model artifacts. |
| `keyring` | `25.7.0` | MIT | Used for Gemini API key storage. |
| `numpy` | `2.4.2` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | Used by the SUP parser and image-processing pipeline. |
| `Pillow` | `12.1.1` | `MIT-CMU` | Used for image decoding, rendering, and OCR image transport. |

### Optional `hf` Extra

These packages are installed only when the optional `hf` extra is requested,
for example with `uv sync --extra hf`.

| Package | Version | License | Notes |
| --- | --- | --- | --- |
| `torch` | `2.10.0` | `BSD-3-Clause` | Required for actual HF OCR inference. Not needed for `istots setup` downloads. |
| `transformers` | `5.1.0` | Apache 2.0 | Loads the HF processor and model used by `--engine hf`. |

## Locked Transitive Dependencies

The following packages appear in `uv.lock` revision `3` as transitive or
platform-conditional dependencies of the current product surface.

### Core / Setup / Auth Path

- `annotated-doc 0.0.4`
- `anyio 4.12.1`
- `backports-tarfile 1.2.0`
- `certifi 2026.1.4`
- `cffi 2.0.0`
- `charset-normalizer 3.4.7`
- `click 8.3.1`
- `cryptography 46.0.7`
- `filelock 3.24.0`
- `fsspec 2026.2.0`
- `h11 0.16.0`
- `hf-xet 1.2.0`
- `httpcore 1.0.9`
- `httpx 0.28.1`
- `idna 3.11`
- `importlib-metadata 9.0.0` (`python < 3.12`)
- `jaraco-classes 3.4.0`
- `jaraco-context 6.1.2`
- `jaraco-functools 4.4.0`
- `jeepney 0.9.0` (`linux`)
- `more-itertools 11.0.2`
- `packaging 26.0`
- `pycparser 3.0`
- `pywin32-ctypes 0.2.3` (`win32`)
- `PyYAML 6.0.3`
- `requests 2.33.1`
- `SecretStorage 3.5.0` (`linux`)
- `shellingham 1.5.4`
- `tqdm 4.67.3`
- `typer 0.23.1`
- `typer-slim 0.23.1`
- `typing-extensions 4.15.0`
- `urllib3 2.6.3`
- `zipp 3.23.0` (`python < 3.12`)

### Optional `hf` Extra Path

- `Jinja2 3.1.6`
- `MarkupSafe 3.0.3`
- `mpmath 1.3.0`
- `networkx 3.6.1`
- `regex 2026.1.15`
- `safetensors 0.7.0`
- `setuptools 70.3.0` (`python >= 3.12`)
- `sympy 1.14.0`
- `tokenizers 0.22.2`
- `triton 3.6.0` (`linux`, `x86_64`)

## CUDA and NVIDIA-Specific Optional Runtime Packages

These packages may be present when a CUDA-enabled PyTorch build is installed.
They are platform-conditional and may be absent from CPU-only deployments.

| Package | Version | Notes |
| --- | --- | --- |
| `cuda-bindings` | `12.9.4` | Linux `x86_64` CUDA-enabled Torch path |
| `cuda-pathfinder` | `1.3.4` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cublas-cu12` | `12.8.4.1` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cuda-cupti-cu12` | `12.8.90` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cuda-nvrtc-cu12` | `12.8.93` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cuda-runtime-cu12` | `12.8.90` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cudnn-cu12` | `9.10.2.21` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cufft-cu12` | `11.3.3.83` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cufile-cu12` | `1.13.1.3` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-curand-cu12` | `10.3.9.90` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cusolver-cu12` | `11.7.3.90` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cusparse-cu12` | `12.5.8.93` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-cusparselt-cu12` | `0.7.1` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-nccl-cu12` | `2.27.5` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-nvjitlink-cu12` | `12.8.93` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-nvshmem-cu12` | `3.4.5` | Linux `x86_64` CUDA-enabled Torch path |
| `nvidia-nvtx-cu12` | `12.8.90` | Linux `x86_64` CUDA-enabled Torch path |

## Development-Only Dependencies

These packages are used for local testing and linting and are not part of the
main product runtime.

| Package | Version | License |
| --- | --- | --- |
| `pytest` | `9.0.2` | MIT |
| `ruff` | `0.15.1` | MIT |
| `iniconfig` | `2.3.0` | MIT |
| `pluggy` | `1.6.0` | MIT |
| `Pygments` | `2.19.2` | `BSD-2-Clause` |
| `colorama` | `0.4.6` | MIT (`win32` only) |

## Redistribution Notes

- If you redistribute an environment containing the optional `hf` extra, also
  include the applicable license files and notices for `torch`,
  `transformers`, and any bundled CUDA/NVIDIA runtime wheels actually present.
- `numpy`, `Pillow`, and `torch` bundle additional third-party license material
  in their wheel metadata. Include those upstream license directories and
  notices when redistributing binary environments that contain them.
- `llama-server` is outside the Python package dependency graph of this
  repository. If you redistribute a host image or installer that bundles a
  `llama-server` binary, you must separately include the upstream `llama.cpp`
  license and notice material for the exact binary build you ship.
- Model artifacts downloaded from Hugging Face are governed by their upstream
  repository and model-card terms. Review the current upstream license before
  mirroring or redistributing downloaded weights or GGUF artifacts.
- If you redistribute a Python runtime that includes `tkinter`, also review the
  Tcl/Tk license material bundled with that Python distribution.
- If `pyproject.toml`, `uv.lock`, the optional `hf` extra, the default model
  repositories, the host-provided `llama-server` binary policy, or the Gemini
  integration surface changes, regenerate this file.
