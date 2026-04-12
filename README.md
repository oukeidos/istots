# IStoTS: Image Subtitles to Text Subtitles

`istots` converts Blu-ray `SUP` subtitles into `SRT` using OCR with
`PaddleOCR-VL-1.5`. Supports preprocessing for masking furigana in Japanese subtitles.

## Requirements

- Python 3.11+
- `uv`

## Setup

```bash
uv sync
```

Install the optional HF fallback runtime only if you plan to use `--engine hf`:

```bash
uv sync --extra hf
```

Download retained runtime assets once:

```bash
uv run istots setup
uv run istots setup --with-qwen-corrector
```

This prepares the retained local setup assets:

- the HF fallback OCR model
- the GGUF OCR runtime model
- the base GGUF mmproj
- the derived `min_pixels=32768` GGUF mmproj
- optionally, the retained local Qwen corrector GGUF and mmproj assets

`uv sync` prepares the retained primary `llama-server` path. The `hf` engine remains available as an explicit optional fallback and requires `uv sync --extra hf`.

`uv run istots setup` covers the retained primary OCR path, the optional faster OCR path, the default detector path, and the explicit HF fallback assets. Add `--with-qwen-corrector` to also provision the retained local Qwen corrector assets from `unsloth/Qwen3.5-35B-A3B-GGUF` using `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` and `mmproj-BF16.gguf`. Gemini credentials are managed separately through `istots auth gemini ...`.

## Quick Start

```bash
uv run istots input.sup output.srt
```

Runtime preflight:

```bash
uv run istots doctor --engine llama-server --role ocr
```

Quick validation on the retained default sample:

```bash
uv run istots smoke
uv run istots smoke --output-dir ./artifacts/smoke
uv run istots smoke --ocr-mode fast
uv run istots smoke --corrector qwen-local
uv run istots smoke --corrector qwen-local --corrector-no-mmproj-offload
uv run istots smoke --corrector qwen-local --corrector-model-path /path/to/qwen.gguf --corrector-mmproj-path /path/to/qwen-mmproj.gguf
uv run istots auth gemini set
uv run istots smoke --corrector gemini
```

Common convert examples:

```bash
uv run istots input.sup output.srt --furigana-mask
uv run istots input.sup output.srt --srt-policy overlap
uv run istots input.sup output.srt --ocr-mode fast
uv run istots input.sup output.srt --detector-output detector.jsonl
uv run istots input.sup output.srt --corrector qwen-local
uv run istots input.sup output.srt --corrector qwen-local --corrector-no-mmproj-offload
uv run istots input.sup output.srt --corrector qwen-local --corrector-model-path /path/to/qwen.gguf --corrector-mmproj-path /path/to/qwen-mmproj.gguf
uv run istots input.sup output.srt --corrector gemini --corrector-output corrected.jsonl
```

For the full CLI surface, run `uv run istots --help`.

Global flags:

- `--help`: show CLI help, including subcommand details.
- `--version`: show the installed `istots` version.

`convert` flags:

- `--engine {llama-server,hf}`: choose the OCR engine. Default is `llama-server`. Use `hf` for the explicit fallback path.
- `--engine hf` requires the optional HF runtime from `uv sync --extra hf`.
- `--hf-device {auto,cpu,gpu}`: HF-only device selection. `auto` prefers CUDA when available and otherwise uses CPU.
- `--hf-dtype {auto,float32,float16,bfloat16}`: HF-only torch dtype policy. `auto` prefers BF16 on supported GPU or CPU paths and otherwise falls back conservatively.
- `--model-id MODEL_ID`: HF model ID or local HF model path for `--engine hf`.
- `--models-dir MODELS_DIR`: local model cache root. Default is `~/.cache/istots/models` or `ISTOTS_MODELS_DIR`.
- `--max-items MAX_ITEMS`: process only the first N subtitle items for debugging.
- `--max-new-tokens MAX_NEW_TOKENS`: maximum generated tokens per subtitle image.
- OCR requests run sequentially, one subtitle image at a time.
- `--ocr-mode {default,fast}`: retained default OCR or the optional faster hybrid OCR path. `fast` uses `ocr-fast` for non-tall rows and retained `ocr` for tall rows.
- `--runtime-profile {auto,cpu}`: retained `llama-server` runtime profile. Default is `auto`.
- `--runtime-profile auto` leaves low-level hardware selection to `llama-server`.
- `--runtime-profile cpu` forces `llama-server` CPU execution.
- `--llama-server-path LLAMA_SERVER_PATH`: explicit `llama-server` binary path.
- `--runtime-port PORT`: override the retained `llama-server` port for convert when `--ocr-mode default` is used.
- `--threads N`: override `llama-server` thread count.
- `--threads-batch N`: override `llama-server` batch thread count.
- `--gpu-layers N`: override `llama-server` GPU layer count.
- `--no-mmproj-offload`: disable `mmproj` offload for `llama-server`.
- `--startup-timeout-sec SECONDS`: `llama-server` startup timeout.
- `--furigana-mask`: enable optional furigana masking before OCR. Default is disabled.
- `--detector-output DETECTOR_OUTPUT`: write retained hybrid detector disagreements as JSONL. Requires `--engine llama-server` with `--ocr-mode default`.
- `--corrector {off,qwen-local,gemini}`: attach the retained conservative anchor-only corrector to `convert`. Requires `--engine llama-server` with `--ocr-mode default`.
- `--corrector-output CORRECTOR_OUTPUT`: optional JSONL path for conservative correction records.
- `--corrector-model-path CORRECTOR_MODEL_PATH`: optional explicit local GGUF corrector model path override for `--corrector qwen-local`.
- `--corrector-mmproj-path CORRECTOR_MMPROJ_PATH`: optional explicit local GGUF corrector mmproj path override for `--corrector qwen-local`.
- `--corrector-port PORT`: override the retained corrector port for `--corrector qwen-local`.
- `--corrector-no-mmproj-offload`: force `--no-mmproj-offload` for `--corrector qwen-local`.
- `--corrector-startup-timeout-sec SECONDS`: startup timeout for `--corrector qwen-local`.
- `--corrector-gemini-model MODEL`: Gemini model id for `--corrector gemini`.
- `--corrector-api-key-env ENV`: environment variable name used when resolving Gemini credentials from the configured `.env` file or the current shell environment.
- `--corrector-thinking-level LEVEL`: optional Gemini thinking level for `--corrector gemini`.
- `--corrector-media-resolution LEVEL`: optional Gemini media resolution level for `--corrector gemini`.
- `--corrector-cache-dir PATH`: optional cache directory for `--corrector gemini`.
- `--srt-policy {safe,overlap}`: SRT output policy. `safe` merges simultaneous windows into one cue. `overlap` keeps overlapping cues separate.
- `--quiet`: suppress progress logs.
- `--force`: overwrite an existing output `.srt` file without prompting.

`setup` flags:

- `--model-id MODEL_ID`: HF fallback model ID to download. Default is `PaddlePaddle/PaddleOCR-VL-1.5`.
- `--gguf-model-id GGUF_MODEL_ID`: GGUF model ID to download. Default is `PaddlePaddle/PaddleOCR-VL-1.5-GGUF`.
- `--with-qwen-corrector`: also download the retained local Qwen corrector assets.
- `--qwen-corrector-model-id MODEL_ID`: Qwen corrector model repository. Default is `unsloth/Qwen3.5-35B-A3B-GGUF`.
- `--qwen-corrector-model-filename FILENAME`: Qwen GGUF filename to download. Default is `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`.
- `--qwen-corrector-mmproj-filename FILENAME`: Qwen mmproj filename to download. Default is `mmproj-BF16.gguf`.
- `--models-dir MODELS_DIR`: local model cache root. Default is `~/.cache/istots/models` or `ISTOTS_MODELS_DIR`.
- `--support-dir SUPPORT_DIR`: local support cache root for pinned gguf snapshot fallback. Default is `~/.cache/istots/support` or `ISTOTS_SUPPORT_DIR`.
- `--gguf-py-base-url GGUF_PY_BASE_URL`: override source root for the pinned gguf snapshot fallback.
- `--gguf-source-mode {auto-download,installed,auto}`: choose whether setup uses an installed pinned `gguf` package or the pinned snapshot fallback.
- `--min-pixels MIN_PIXELS`: `clip.vision.image_min_pixels` value for the derived GGUF mmproj. Default is `32768`.
- `--force`: re-download and re-materialize even when local assets already exist.
- `--quiet`: suppress progress logs.

Optional environment variables:

```bash
# Override model cache root
export ISTOTS_MODELS_DIR="$HOME/.cache/istots/models"

# Override support cache root for pinned gguf snapshot fallback
export ISTOTS_SUPPORT_DIR="$HOME/.cache/istots/support"

# Override the local auth config path that stores the configured Gemini .env file path
export ISTOTS_AUTH_CONFIG_PATH="$HOME/.config/istots/auth.json"
```

## Gemini Auth

Use `auth gemini` to manage Gemini credentials without printing the API key in the terminal.

```bash
uv run istots auth gemini set
uv run istots auth gemini delete
uv run istots auth gemini status
uv run istots auth gemini env-file set /path/to/.env
uv run istots auth gemini env-file clear
```

Credential resolution order for `--corrector gemini` is:

1. local keyring
2. the configured `.env` file path
3. the current shell environment

Recommended `.env` template:

```dotenv
GEMINI_API_KEY=your-gemini-api-key-here
```

`GEMINI_API_KEY` is the standard key name. `GOOGLE_API_KEY` remains accepted as a compatibility alias when the default key name is in use.

## Quick Validation

Use `smoke` when you want the retained fast regression path instead of a full manual convert command.

- `smoke` defaults to `../test/sample.sup`, which is the retained minimum sample for required smoke tests.
- If `--output-dir` is omitted, `istots` writes smoke artifacts to a temporary directory.
- The default smoke path writes a smoke SRT plus the retained hybrid detector manifest.
- `uv run istots smoke --ocr-mode fast` exercises the optional faster OCR path on the same sample.
- `--corrector qwen-local` or `--corrector gemini` adds a correction manifest alongside the smoke SRT.

Recommended order for a new machine or runtime profile:

1. `uv sync`
2. `uv run istots setup`
3. `uv run istots doctor --engine llama-server --role ocr`
4. `uv run istots smoke`
5. `uv run istots input.sup output.srt`

## Runtime Doctor

Use `doctor` before switching to retained `llama-server` runtime roles:

- `uv run istots doctor --engine llama-server --role ocr`
- `uv run istots doctor --engine llama-server --role ocr-fast --profile cpu`
- `uv run istots doctor --engine llama-server --role detector`

The doctor checks:

- `llama-server` binary presence
- required model and mmproj assets for the selected role
- likely port conflicts
- launch readiness
- minimal OpenAI-compatible smoke response

## Runtime Profiles

- `auto`: the default retained profile. Use this first on supported GPU hosts or when you want `llama-server` to choose the lowest-level launch details.
- `cpu`: the official force-CPU profile for hosts without a usable GPU path or when you want a deterministic CPU-only run.

Advanced `llama-server` overrides remain available on `convert`, `doctor`, and `smoke` through:

- `--runtime-profile`
- `--runtime-port`
- `--threads`
- `--threads-batch`
- `--gpu-layers`
- `--no-mmproj-offload`
- `--llama-server-path`

## Host Patterns

- GPU-capable host: start with the default `auto` profile and let `llama-server` choose its own hardware path.
- CPU-only host: use `--runtime-profile cpu`.

## OCR Modes

- `default`: the retained primary OCR path. All rows use the retained `ocr` runtime role.
- `fast`: the retained optional faster OCR path. `istots` partitions rows by image ratio, sends non-tall rows to `ocr-fast`, sends tall rows to retained `ocr`, and restores the original row order before SRT assembly.

## Detector Output

- `--detector-output`: runs the retained hybrid detector alongside the retained default OCR path and writes disagreement rows as JSONL.
- Non-tall rows use the `alternate_read_non_tall` branch backed by `ocr-fast`.
- Tall rows use the `repeat_drift_tall` branch backed by the retained `detector` role.
- In the first-wave product surface, detector behavior is retained on the `llama-server` path and is not mirrored onto the explicit `hf` fallback path.

## Conservative Correction

- Correction remains convert-attached, opt-in, and conservative anchor-only.
- The retained hybrid detector disagreement surface is the default correction trigger surface.
- `--corrector qwen-local` uses the retained `strict_ocr_v1` prompt with the retained local Qwen runtime recipe.
- If the retained local Qwen corrector assets were provisioned with `uv run istots setup --with-qwen-corrector`, `--corrector qwen-local` can run without explicit model or mmproj path overrides.
- `--corrector-no-mmproj-offload` is available as an opt-in Qwen local override and is not forced by default.
- `--corrector gemini` uses `strict_ocr_v1` on non-tall rows and adds `general_vertical_hint_v1` on tall rows.
- `uv run istots auth gemini set` stores the Gemini API key in the local keyring, and `uv run istots auth gemini env-file set /path/to/.env` configures the fallback `.env` path.

## HF Fallback

- `llama-server` remains the primary OCR path.
- `hf` remains an explicit optional fallback engine rather than an automatic routing mode.
- The `hf` engine requires the optional HF runtime: `uv sync --extra hf`.

## Language Support

The default OCR model is multilingual. PaddleOCR officially documents the
PaddleOCR-VL model series as supporting 109 languages.

For the full official language coverage list, see:
https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html

## Japanese Furigana Handling

- This is a Japanese-specific text cleanup issue.
- `istots` provides an optional pre-OCR furigana masking mode via `--furigana-mask`.
- The furigana masking path is heuristic and disabled by default.
- Recommended workflow: compare OCR results with and without `--furigana-mask` on your subtitle set.
