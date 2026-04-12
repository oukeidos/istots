# Changelog

## [0.3.0] - 2026-04-12
- Rebuilt the product around a primary `llama-server` OCR path, with `hf` kept as an explicit optional fallback.
- Added retained setup/materialization for Paddle GGUF assets, a derived `min_pixels=32768` mmproj, and optional local Qwen corrector assets.
- Added fast OCR, detector, and conservative correction flows, including local Qwen and Gemini-based correction options.
- Added smoke, structured doctor, and Gemini auth commands for setup, runtime, and workflow validation.
- Simplified runtime behavior around sequential single-backend execution and model-family runtime overrides for PaddleOCR-VL and Qwen3.5.

## [0.2.0] - 2026-04-05
- Added optional furigana masking before OCR to reduce furigana noise in generated subtitles.
- Implemented a multi-window PGS subtitle pipeline with window-aware OCR, debug/export tooling, GUI inspection, and selectable `safe`/`overlap` SRT output policies.

## [0.1.0] - 2026-04-05
- Initial release
