# Scripts

Local helper scripts for debugging and inspection live in this directory.

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
uv run python scripts/image_compare_gui/compare_images.py \
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
