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

Download model once:

```bash
uv run istots setup
```

This prepares the retained local setup assets:

- the HF fallback OCR model
- the GGUF OCR runtime model
- the base GGUF mmproj
- the derived `min_pixels=32768` GGUF mmproj

## Usage

```bash
uv run istots input.sup output.srt
```

Runtime preflight:

```bash
uv run istots doctor --engine llama-server --role ocr
```

Optional flags:

```bash
uv run istots input.sup output.srt --furigana-mask
uv run istots input.sup output.srt --srt-policy overlap
uv run istots input.sup output.srt --ocr-mode fast
uv run istots input.sup output.srt --detector-output detector.jsonl
```

Global flags:

- `--help`: show CLI help, including subcommand details.
- `--version`: show the installed `istots` version.

`convert` flags:

- `--engine {llama-server,hf}`: choose the OCR engine. Default is `llama-server`. Use `hf` for the explicit fallback path.
- `--device {auto,cpu,gpu}`: choose the inference device. `auto` prefers GPU and falls back to CPU.
- `--model-id MODEL_ID`: HF model ID or local HF model path for `--engine hf`.
- `--models-dir MODELS_DIR`: local model cache root. Default is `~/.cache/istots/models` or `ISTOTS_MODELS_DIR`.
- `--max-items MAX_ITEMS`: process only the first N subtitle items for debugging.
- `--max-new-tokens MAX_NEW_TOKENS`: maximum generated tokens per subtitle image.
- `--batch-size BATCH_SIZE`: OCR batch size. Default is `1`. If GPU OOM occurs, `istots` reduces the batch size and retries.
- `--ocr-mode {default,fast}`: retained default OCR or the optional faster hybrid OCR path. `fast` uses `ocr-fast` for non-tall rows and retained `ocr` for tall rows.
- `--runtime-profile {auto,cpu,memory}`: retained `llama-server` runtime profile. Default is `auto`.
- `--llama-server-path LLAMA_SERVER_PATH`: explicit `llama-server` binary path.
- `--runtime-port PORT`: override the retained `llama-server` port for convert when `--ocr-mode default` is used.
- `--threads N`: override `llama-server` thread count.
- `--threads-batch N`: override `llama-server` batch thread count.
- `--gpu-layers N`: override `llama-server` GPU layer count.
- `--no-mmproj-offload`: disable `mmproj` offload for `llama-server`.
- `--startup-timeout-sec SECONDS`: `llama-server` startup timeout.
- `--furigana-mask`: enable optional furigana masking before OCR. Default is disabled.
- `--detector-output DETECTOR_OUTPUT`: write retained hybrid detector disagreements as JSONL. Requires `--engine llama-server` with `--ocr-mode default`.
- `--srt-policy {safe,overlap}`: SRT output policy. `safe` merges simultaneous windows into one cue. `overlap` keeps overlapping cues separate.
- `--quiet`: suppress progress logs.
- `--force`: overwrite an existing output `.srt` file without prompting.

`setup` flags:

- `--model-id MODEL_ID`: HF fallback model ID to download. Default is `PaddlePaddle/PaddleOCR-VL-1.5`.
- `--gguf-model-id GGUF_MODEL_ID`: GGUF model ID to download. Default is `PaddlePaddle/PaddleOCR-VL-1.5-GGUF`.
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
```

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

## OCR Modes

- `default`: the retained primary OCR path. All rows use the retained `ocr` runtime role.
- `fast`: the retained optional faster OCR path. `istots` partitions rows by image ratio, sends non-tall rows to `ocr-fast`, sends tall rows to retained `ocr`, and restores the original row order before SRT assembly.

## Detector Output

- `--detector-output`: runs the retained hybrid detector alongside the retained default OCR path and writes disagreement rows as JSONL.
- Non-tall rows use the `alternate_read_non_tall` branch backed by `ocr-fast`.
- Tall rows use the `repeat_drift_tall` branch backed by the retained `detector` role.

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
