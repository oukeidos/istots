# Image Compare GUI

Simple multi-panel GUI for visually comparing aligned images from separate directories.

## Usage

```bash
python tools/image_compare_gui/compare_images.py \
  --panel original=out/original \
  --panel masked=out/masked
```

Three-way comparison also works:

```bash
python tools/image_compare_gui/compare_images.py \
  --panel original=out/original \
  --panel overlay=out/overlay \
  --panel final=out/final
```

Files are matched by relative path under each directory.

By default, the viewer uses the union of available files, so missing files are shown as missing panels.
Use `--intersection-only` to compare only files present in every panel.

When viewing furigana debug exports, the GUI auto-detects `manifest.jsonl` from the shared parent
directory and exposes a `Masking Only` toggle button inside the window.
When the manifest includes `frame_index`, `window_id`, `bbox`, and time fields, the GUI also:

- keeps manifest order instead of naive filename sorting
- shows frame/window/bbox/time metadata in the toolbar
- lets you move by frame group with `Prev Frame`, `Next Frame`, `PageUp`, and `PageDown`
- keeps all segments from a masked frame when `Masking Only` is enabled

```bash
python tools/image_compare_gui/compare_images.py \
  --panel original=out_debug/original \
  --panel lines=out_debug/lines \
  --panel masked=out_debug/masked \
  --panel mask=out_debug/mask
```

## Keys

- `Left` / `Right`: previous and next image
- `A` / `D`: previous and next image
- `PageUp` / `PageDown`: previous and next frame group
- `Segment` input + `Jump`: move directly to a specific segment number
- `+` / `-`: zoom in and out
- `0`: reset to fit
- `Home` / `End`: first and last image
- `Q`: quit
