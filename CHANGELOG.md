# Changelog

## [0.3.2] - 2026-04-14
- Applied a narrow one-shot fallback mitigation for main-OCR token-limit truncation issues in both `llama-server` and HF backends while leaving detector and corrector behavior unchanged.

## [0.3.1] - 2026-04-14
- Improved the PGS parser by tightening composition-state handling, fixing full-surface multi-window composition, sharing frame assembly across parser and `sup_reader`, and adding an OCR-facing path that skips unnecessary composed-surface work.
- Improved furigana-mask throughput by reusing duplicate-image analyses within a SUP batch, adding process-based batch parallel analysis, switching threshold estimation to histogram-backed computation, replacing per-pixel component extraction with row-run connected components plus array-backed union-find, and moving line construction to slab/band scans while preserving output behavior.

## [0.3.0] - 2026-04-13
- Rebuilt the product around a primary `llama-server` OCR path, with `hf` kept as an explicit optional fallback.
- Added retained setup/materialization for Paddle GGUF assets, a derived `min_pixels=32768` mmproj, and optional local Qwen corrector assets.
- Added fast OCR, detector, and conservative correction flows, including local Qwen and Gemini-based correction options.
- Added smoke, structured doctor, and Gemini auth commands for setup, runtime, and workflow validation.

## [0.2.0] - 2026-04-05
- Added optional furigana masking before OCR to reduce furigana noise in generated subtitles.
- Implemented a multi-window PGS subtitle pipeline with window-aware OCR, debug/export tooling, GUI inspection, and selectable `safe`/`overlap` SRT output policies.

## [0.1.0] - 2026-04-05
- Initial release
