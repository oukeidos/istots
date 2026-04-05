# Third-Party Notices

This document records the third-party software and model artifacts used by
`istots` in the current repository state.

## Inventory Basis

This notice file was prepared from:

- `pyproject.toml`
- `uv.lock` (revision `3`)
- installed Python distribution metadata (`*.dist-info`)
- source inspection of `src/istots/*`
- the model card for `PaddlePaddle/PaddleOCR-VL-1.5` on Hugging Face

Important scope notes:

- The exact resolved dependency set can vary by Python version, operating
  system, architecture, and whether PyTorch is installed in CPU-only or
  CUDA-enabled form.
- `pyproject.toml` declares `uv_build` as the build backend. It is a build-time
  dependency, not part of the main runtime dependency set, so it is not
  enumerated below.
- License strings below are taken from installed package metadata unless noted
  otherwise. For binary redistribution, the bundled upstream license files take
  precedence over shortened metadata labels.

## First-Party License

- `istots`: MIT. See `LICENSE`.

## External Model Artifact

`istots` is configured to use the following external model by default:

| Component | Source | License | Notes |
| --- | --- | --- | --- |
| `PaddlePaddle/PaddleOCR-VL-1.5` | <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5> | `apache-2.0` | Referenced by `src/istots/model_store.py` as `DEFAULT_MODEL_ID`. Model files are downloaded by `istots setup` into a local cache and are not vendored in this repository. |

## Direct Runtime Dependencies

These packages are declared directly in `pyproject.toml`.

| Package | Version | License | Usage in `istots` |
| --- | --- | --- | --- |
| `huggingface-hub` | `1.4.1` | Apache | Used by `istots setup` to download the OCR model from Hugging Face. |
| `numpy` | `2.4.2` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | Required by the OCR/model stack. |
| `pillow` | `12.1.1` | `MIT-CMU` | Used for image handling in the SUP parser and OCR backend. |
| `torch` | `2.10.0` | `BSD-3-Clause` | Provides inference runtime and CUDA detection. |
| `transformers` | `5.1.0` | Apache 2.0 License | Loads the processor and model used for OCR inference. |

## Transitive Runtime Dependencies

These packages are present in the locked runtime dependency graph used to
prepare this notice.

| Package | Version | License |
| --- | --- | --- |
| `annotated-doc` | `0.0.4` | `MIT` |
| `anyio` | `4.12.1` | `MIT` |
| `certifi` | `2026.1.4` | `MPL-2.0` |
| `click` | `8.3.1` | `BSD-3-Clause` |
| `filelock` | `3.24.0` | `MIT` |
| `fsspec` | `2026.2.0` | `BSD-3-Clause` |
| `h11` | `0.16.0` | `MIT` |
| `hf-xet` | `1.2.0` | `Apache-2.0` |
| `httpcore` | `1.0.9` | `BSD-3-Clause` |
| `httpx` | `0.28.1` | `BSD-3-Clause` |
| `idna` | `3.11` | `BSD-3-Clause` |
| `jinja2` | `3.1.6` | BSD license (see bundled `LICENSE.txt`) |
| `markdown-it-py` | `4.0.0` | `MIT` |
| `markupsafe` | `3.0.3` | `BSD-3-Clause` |
| `mdurl` | `0.1.2` | `MIT` |
| `mpmath` | `1.3.0` | BSD license |
| `networkx` | `3.6.1` | `BSD-3-Clause` |
| `packaging` | `26.0` | `Apache-2.0 OR BSD-2-Clause` |
| `pygments` | `2.19.2` | `BSD-2-Clause` |
| `pyyaml` | `6.0.3` | `MIT` |
| `regex` | `2026.1.15` | `Apache-2.0 AND CNRI-Python` |
| `rich` | `14.3.2` | `MIT` |
| `safetensors` | `0.7.0` | Apache Software License |
| `setuptools` | `70.3.0` | `MIT` |
| `shellingham` | `1.5.4` | ISC License |
| `sympy` | `1.14.0` | BSD license |
| `tokenizers` | `0.22.2` | Apache Software License |
| `tqdm` | `4.67.3` | `MPL-2.0 AND MIT` |
| `triton` | `3.6.0` | `MIT` |
| `typer` | `0.23.1` | `MIT` |
| `typer-slim` | `0.23.1` | `MIT` |
| `typing-extensions` | `4.15.0` | `PSF-2.0` |

## CUDA and NVIDIA-Specific Runtime Packages

The following packages may be present when using a CUDA-enabled PyTorch build.
They may be absent from CPU-only deployments.

Where NVIDIA package metadata is mixed or conservative, this document preserves
the installed metadata labels and treats those wheels as vendor-governed.
Review the bundled NVIDIA license files before redistribution.

| Package | Version | License |
| --- | --- | --- |
| `cuda-bindings` | `12.9.4` | `LicenseRef-NVIDIA-SOFTWARE-LICENSE` |
| `cuda-pathfinder` | `1.3.4` | `Apache-2.0` |
| `nvidia-cublas-cu12` | `12.8.4.1` | NVIDIA Proprietary Software |
| `nvidia-cuda-cupti-cu12` | `12.8.90` | NVIDIA Proprietary Software |
| `nvidia-cuda-nvrtc-cu12` | `12.8.93` | NVIDIA Proprietary Software |
| `nvidia-cuda-runtime-cu12` | `12.8.90` | NVIDIA Proprietary Software |
| `nvidia-cudnn-cu12` | `9.10.2.21` | `LicenseRef-NVIDIA-Proprietary` |
| `nvidia-cufft-cu12` | `11.3.3.83` | NVIDIA Proprietary Software |
| `nvidia-cufile-cu12` | `1.13.1.3` | NVIDIA Proprietary Software |
| `nvidia-curand-cu12` | `10.3.9.90` | NVIDIA Proprietary Software |
| `nvidia-cusolver-cu12` | `11.7.3.90` | NVIDIA Proprietary Software |
| `nvidia-cusparse-cu12` | `12.5.8.93` | NVIDIA Proprietary Software |
| `nvidia-cusparselt-cu12` | `0.7.1` | NVIDIA Proprietary Software |
| `nvidia-nccl-cu12` | `2.27.5` | `BSD-3-Clause` license text bundled with the wheel |
| `nvidia-nvjitlink-cu12` | `12.8.93` | NVIDIA Proprietary Software |
| `nvidia-nvshmem-cu12` | `3.4.5` | `LicenseRef-NVIDIA-Proprietary` |
| `nvidia-nvtx-cu12` | `12.8.90` | Apache 2.0 |

## Development-Only Dependencies

These packages are installed for local testing and linting. They are not part
of the main runtime path of `istots`.

| Package | Version | License |
| --- | --- | --- |
| `pytest` | `9.0.2` | `MIT` |
| `ruff` | `0.15.1` | `MIT` |
| `iniconfig` | `2.3.0` | `MIT` |
| `pluggy` | `1.6.0` | `MIT` |

## Redistribution Notes

- `numpy`, `pillow`, and `torch` ship bundled third-party license material in
  their installed wheel metadata. If you redistribute those wheels, or a binary
  image containing them, include the corresponding upstream license directories
  and notices from the installed distributions.
- In particular, check the `licenses/` material bundled in the installed
  distributions for `numpy`, `pillow`, and `torch`.
- `torch` distributions may also ship an additional upstream `NOTICE` file.
- NVIDIA CUDA wheels are not covered by the project's MIT license. Review each
  bundled `License.txt` before mirroring or redistributing those artifacts.
- If `pyproject.toml`, `uv.lock`, the Python version, the platform, or the
  default OCR model changes, regenerate this file.
