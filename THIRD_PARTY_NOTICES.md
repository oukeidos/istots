# Third-Party Notices

This notice accompanies the public `istots` source release. It identifies the
third-party software, optional extras, build-time packaging tools, downloaded
model artifacts, host runtimes, and remote services that may be used with the
current `istots` product surface.

The exact third-party set varies by operating system, Python version,
architecture, and whether the optional `hf` and `gui` extras are installed.

## Notice Basis

This notice reflects the current repository state based on:

- `pyproject.toml`
- `uv.lock`
- the `istots` source tree at version `0.4.3`

## First-Party Software

- `istots` is licensed under the MIT License. See `LICENSE`.

## Direct Python Dependencies Declared By This Project

### Core Runtime

| Package | Locked Version | License | Purpose |
| --- | --- | --- | --- |
| `gguf` | `0.18.0` | MIT | Reads and materializes GGUF mmproj artifacts. |
| `huggingface-hub` | `1.4.1` | Apache | Downloads model artifacts used by setup flows. |
| `keyring` | `25.7.0` | MIT | Stores Gemini API credentials locally. |
| `numpy` | `2.4.2` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | Supports parsing and image-processing logic. |
| `Pillow` | `12.1.1` | `MIT-CMU` | Handles image decoding, rendering, and OCR image transport. |

### Optional `hf` Extra

| Package | Locked Version | License | Purpose |
| --- | --- | --- | --- |
| `torch` | `2.10.0` | `BSD-3-Clause` | Provides the optional Hugging Face OCR runtime. |
| `transformers` | `5.1.0` | `Apache 2.0` | Loads the optional Hugging Face OCR model and processor. |

### Optional `gui` Extra

| Package | Locked Version | License | Purpose |
| --- | --- | --- | --- |
| `PySide6` | `6.11.0` | `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only` | Provides the optional desktop GUI distributed from the Python package surface. |

The current `gui` lock also resolves the following Qt for Python companion
packages:

- `PySide6-Addons 6.11.0`
- `PySide6-Essentials 6.11.0`
- `shiboken6 6.11.0`

These companion packages use the same Qt for Python license expression as
`PySide6`.

## Locked Supporting Packages

The following packages are currently pinned in `uv.lock` as supporting
dependencies of the declared product surface. They are not all present in every
installation.

### Base Runtime, Setup, Auth, and CLI-Adjacent Support

- `annotated-doc 0.0.4`
- `anyio 4.12.1`
- `backports-tarfile 1.2.0` (`python < 3.12`)
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
- `markdown-it-py 4.0.0`
- `mdurl 0.1.2`
- `more-itertools 11.0.2`
- `packaging 26.0`
- `pycparser 3.0`
- `PyYAML 6.0.3`
- `pywin32-ctypes 0.2.3` (`win32`)
- `requests 2.33.1`
- `rich 14.3.2`
- `SecretStorage 3.5.0` (`linux`)
- `shellingham 1.5.4`
- `tqdm 4.67.3`
- `typer 0.23.1`
- `typer-slim 0.23.1`
- `typing-extensions 4.15.0`
- `urllib3 2.6.3`
- `zipp 3.23.0` (`python < 3.12`)

### Optional `hf` Extra Support

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

### Optional CUDA and NVIDIA Runtime Packages

These packages may be present when a CUDA-enabled PyTorch build is installed:

- `cuda-bindings 12.9.4`
- `cuda-pathfinder 1.3.4`
- `nvidia-cublas-cu12 12.8.4.1`
- `nvidia-cuda-cupti-cu12 12.8.90`
- `nvidia-cuda-nvrtc-cu12 12.8.93`
- `nvidia-cuda-runtime-cu12 12.8.90`
- `nvidia-cudnn-cu12 9.10.2.21`
- `nvidia-cufft-cu12 11.3.3.83`
- `nvidia-cufile-cu12 1.13.1.3`
- `nvidia-curand-cu12 10.3.9.90`
- `nvidia-cusolver-cu12 11.7.3.90`
- `nvidia-cusparse-cu12 12.5.8.93`
- `nvidia-cusparselt-cu12 0.7.1`
- `nvidia-nccl-cu12 2.27.5`
- `nvidia-nvjitlink-cu12 12.8.93`
- `nvidia-nvshmem-cu12 3.4.5`
- `nvidia-nvtx-cu12 12.8.90`

### Source Release Build, Test, and Lint Tools

These tools are used for source packaging, verification, and local release
authoring. They are not part of the normal end user runtime as standalone
application dependencies:

| Package | Locked Version | License |
| --- | --- | --- |
| `PyInstaller` | `6.19.0` | `GPL-2.0` with the PyInstaller bootloader exception, plus `Apache-2.0` for certain files |
| `pyinstaller-hooks-contrib` | `2026.4` | `GPL-2.0-or-later` and `Apache-2.0`, depending on the included hook files |
| `pytest` | `9.0.2` | MIT |
| `ruff` | `0.15.1` | MIT |
| `iniconfig` | `2.3.0` | MIT |
| `pluggy` | `1.6.0` | MIT |
| `Pygments` | `2.19.2` | `BSD-2-Clause` |
| `colorama` | `0.4.6` | MIT (`win32` only) |

## Host-Provided Build Tools, Runtimes, and Remote Services

| Component | Source | License / Terms | Notes |
| --- | --- | --- | --- |
| Inno Setup compiler (`ISCC.exe`) | Local Windows build host tool obtained from the `jrsoftware/issrc` project or JRSoftware downloads | The Inno Setup license applies. JRSoftware also publishes separate commercial-license and purchasing terms for organizations that use the compiler in commercial contexts. | Used only to compile the optional Windows installer. The compiler itself is not bundled into the shipped app or installer payload. |
| `llama-server` runtime binary | User-provided binary or GUI-managed install obtained from the `llama.cpp` project | The license and notices for the exact upstream `llama.cpp` build you ship, install, or download apply | Required for the primary OCR path, the fast OCR path, and the local Qwen corrector path. This repository does not vendor the binary. The Windows GUI can install a managed copy from official upstream releases. |
| GitHub Releases API (`ggml-org/llama.cpp`) | Remote GitHub service | GitHub service terms apply | Used by the Windows GUI managed-runtime setup flow to resolve the latest official Windows runtime release metadata. |
| Google Generative Language API (`Gemini`) | Remote Google service | Google API and model terms apply | Used only when the Gemini corrector path is enabled. No Gemini SDK or model weights are redistributed by this repository. |

## Downloaded Runtime, Model, and Artifact Sources

`istots` does not vendor the following runtime files, installers, model files,
or derived artifacts in the repository. They are downloaded or materialized
locally when the relevant setup flows are used.

| Artifact | Source | License / Terms | Notes |
| --- | --- | --- | --- |
| Official Windows `llama.cpp` runtime archives | <https://github.com/ggml-org/llama.cpp/releases> | Review the exact upstream release terms and notices for the downloaded build | Used only by the optional Windows GUI managed-runtime setup flow. Installed locally as a managed runtime, not vendored in this repository. |
| Microsoft Visual C++ Redistributable (x64) installer | <https://aka.ms/vs/17/release/vc_redist.x64.exe> | Microsoft license terms for the downloaded installer apply | Downloaded only when the optional Windows GUI managed-runtime setup flow is allowed to install missing Windows prerequisites. |
| HF fallback OCR model | <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5> | `apache-2.0` according to the upstream model card | Used only by the optional `--engine hf` path. |
| Primary OCR GGUF model and base mmproj | <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-GGUF> | Review the upstream repository and model-card terms at the time of download | Used by the retained primary local OCR path. |
| Derived fast OCR mmproj | Locally materialized from the official Paddle GGUF base mmproj | Inherits the applicable terms of the upstream source artifact | Created locally by `istots`; not downloaded as a separate upstream file. |
| Optional local Qwen corrector model and mmproj | <https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF> | Review the upstream repository and model-card terms at the time of download | Used only when the optional local Qwen corrector assets are installed. |
| Optional pinned `gguf-py` snapshot fallback | <https://github.com/ggml-org/llama.cpp/tree/94ca829b6001019622c0f67fcd48e9ec6bd7dce8/gguf-py> | MIT | Used only as a fallback source when the installed `gguf` package is unavailable and the configured source mode allows downloading it. |

## Redistribution Guidance

- If you distribute `istots` with the optional desktop GUI enabled, include the
  applicable notices for `PySide6`, `PySide6-Addons`, `PySide6-Essentials`, and
  `shiboken6`, and comply with the Qt for Python license terms you rely on.
- If you distribute `istots` with the optional `hf` extra or with CUDA-enabled
  PyTorch wheels, include the applicable notices for `torch`,
  `transformers`, and any CUDA or NVIDIA runtime packages actually bundled.
- If you distribute packaged Windows GUI bundles generated by `PyInstaller`,
  review the current PyInstaller and `pyinstaller-hooks-contrib` license terms
  used in your build workflow, together with the obligations of the bundled
  dependencies that the frozen app actually ships.
- If you distribute Windows installers built with Inno Setup, review and comply
  with the current Inno Setup license and any applicable compiler-use terms for
  your distribution context.
- If you distribute a package, installer, or image that bundles a
  `llama-server` binary or a GUI-managed Windows runtime archive, you must
  separately include the license and notice material for the exact `llama.cpp`
  build that you ship.
- If you mirror or redistribute downloaded model weights, GGUF artifacts,
  Windows runtime archives, or Microsoft prerequisite installers, review and
  comply with the current upstream repository, model-card, and installer terms
  for those assets.
- Some wheels, including `numpy`, `Pillow`, `torch`, and Qt for Python wheels,
  may contain additional upstream license files or notice material. Preserve
  those bundled notices when redistributing binary environments built from
  those wheels.
- Refresh this notice whenever `pyproject.toml`, `uv.lock`, the optional extras,
  bundled runtimes, or model sources change in a way that affects third-party
  obligations.
