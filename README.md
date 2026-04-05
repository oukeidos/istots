# IStoTS: Image Subtitles to Text Subtitles

`istots` converts Blu-ray `SUP` subtitles into `SRT` using OCR with
`PaddlePaddle/PaddleOCR-VL-1.5`.

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

## Usage

```bash
uv run istots input.sup output.srt
```

Optional flags:

```bash
uv run istots input.sup output.srt --device auto --batch-size 8 --model-id PaddlePaddle/PaddleOCR-VL-1.5 --max-items 100 --force
uv run istots setup --force --quiet
```

Global flags:

- `--help`: show CLI help, including subcommand details.
- `--version`: show the installed `istots` version.

`convert` flags:

- `--device {auto,cpu,cuda}`: choose the inference device. `auto` prefers CUDA and falls back to CPU.
- `--model-id MODEL_ID`: model ID or local model path. If a model ID is given, it must already exist in local cache from `istots setup`.
- `--models-dir MODELS_DIR`: local model cache root. Default is `~/.cache/istots/models` or `ISTOTS_MODELS_DIR`.
- `--max-items MAX_ITEMS`: process only the first N subtitle items for debugging.
- `--max-new-tokens MAX_NEW_TOKENS`: maximum generated tokens per subtitle image.
- `--batch-size BATCH_SIZE`: OCR batch size. Default is `1`. If CUDA OOM occurs, `istots` reduces the batch size and retries.
- `--quiet`: suppress progress logs.
- `--force`: overwrite an existing output `.srt` file without prompting.

`setup` flags:

- `--model-id MODEL_ID`: model ID to download. Default is `PaddlePaddle/PaddleOCR-VL-1.5`.
- `--models-dir MODELS_DIR`: local model cache root. Default is `~/.cache/istots/models` or `ISTOTS_MODELS_DIR`.
- `--force`: re-download even when the local cache already exists.
- `--quiet`: suppress progress logs.

Optional environment variables:

```bash
# Override model cache root
export ISTOTS_MODELS_DIR="$HOME/.cache/istots/models"
```

## Language Support

The default OCR model is multilingual. PaddleOCR officially documents the
PaddleOCR-VL model series as supporting 109 languages.

For the full official language coverage list, see:
https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html

## Japanese Furigana Handling

- This is a Japanese-specific text cleanup issue.
- Main `istots` conversion path does not include furigana removal.
- Recommended workflow: run OCR with `istots`, then apply furigana cleanup for Japanese output as a post-processing step.

## Export Deduplicated SUP Images

To compare source subtitle images with SRT output, export deduplicated SUP frames:

```bash
uv run python scripts/export_sup_images.py input.sup out_images
```

This writes:

- PNG files: `out_images/images/000001.png`, ...
- metadata: `out_images/manifest.jsonl` (`index`, `raw_index`, `start_ms`, `end_ms`, `image`)

Optional flags:

- `--max-items N`: export only first N deduplicated frames.
- `--force`: overwrite existing output directory.
