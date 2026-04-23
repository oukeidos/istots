# Scripts

Local helper scripts for debugging and inspection live in this directory.

## Update Windows Runtime Allowlist

The Windows managed-runtime allowlist updater is a standalone maintenance tool.
It scans recent upstream `llama.cpp` releases on a Windows host, records local
validation results in a gitignored artifact directory, and can then apply the
top passing candidates into the committed allowlist source file.

Full usage notes live in:

- `scripts/update_windows_runtime_allowlist.README.md`

Scan recent releases and stop early once the configured per-family target is
met:

```bash
uv run python scripts/update_windows_runtime_allowlist.py scan
```

Apply the top pending passing tags into
`src/istots/gui/windows_runtime_allowlist.py`:

```bash
uv run python scripts/update_windows_runtime_allowlist.py apply
```

Useful knobs:

- `--target x64/cpu=3`: how many new passing tags to collect or apply per
  family for that run
- `--attempt-budget x64/vulkan=6`: max local validation attempts per family in
  a scan run
- `--lookback-days 120`: how far back a scan may look when the last run was a
  long time ago
- `--release-limit 40`: hard cap on how many recent upstream tags one scan
  inspects

By default the tool writes its persistent ledger and latest summaries under
`build/windows_runtime_allowlist_automation/`. Those artifacts are local-only
and gitignored.

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
