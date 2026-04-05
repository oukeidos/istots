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

## Usage

```bash
uv run istots input.sup output.srt
```

Optional flags:

```bash
uv run istots input.sup output.srt --furigana-mask
uv run istots input.sup output.srt --srt-policy overlap
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
- `--furigana-mask`: enable optional furigana masking before OCR. Default is disabled.
- `--srt-policy {safe,overlap}`: SRT output policy. `safe` merges simultaneous windows into one cue. `overlap` keeps overlapping cues separate.
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
- `istots` provides an optional pre-OCR furigana masking mode via `--furigana-mask`.
- The furigana masking path is heuristic and disabled by default.
- Recommended workflow: compare OCR results with and without `--furigana-mask` on your subtitle set.

## Export Furigana Debug Images

To inspect the masking results frame by frame:

```bash
uv run python scripts/export_furigana_debug_images.py input.sup out_debug
```

This writes:

- original subtitle window images: `out_debug/original/000001_w0.png`, ...
- masked images: `out_debug/masked/000001_w0.png`, ...
- binary mask images: `out_debug/mask/000001_w0.png`, ...
- line debug overlays: `out_debug/lines/000001_w0.png`, ...
- metadata: `out_debug/manifest.jsonl`

Line debug overlay colors:

- blue box: main-text line
- red box: furigana line
- yellow box: unclassified line

To compare outputs visually in the GUI:

```bash
uv run python tools/image_compare_gui/compare_images.py \
  --panel original=out_debug/original \
  --panel masked=out_debug/masked \
  --panel lines=out_debug/lines \
  --panel mask=out_debug/mask
```

## Export Deduplicated SUP Images

To compare source subtitle images with SRT output, export deduplicated SUP window images:

```bash
uv run python scripts/export_sup_images.py input.sup out_images
```

This writes:

- PNG files: `out_images/images/000001_w0.png`, ...
- metadata: `out_images/manifest.jsonl`
  (`index`, `frame_index`, `segment_index`, `raw_index`, `window_id`, `bbox`, `start_ms`, `end_ms`, `image`)

Optional flags:

- `--max-items N`: export only first N deduplicated subtitle window inputs.
- `--force`: overwrite existing output directory.
