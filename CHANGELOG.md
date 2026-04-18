# Changelog

## [0.3.4] - 2026-04-18
- Started the retained EXP-008 stack port by promoting PaddleOCR-VL llama-server defaults to `ctx-size=2048`, adding a Paddle-only request-budget restart policy of `200`, and preserving the separate local Qwen `ctx-size=4096` policy.
- Ported the retained parser/pipeline memory seam by releasing parser predecode workers after frame collection, defaulting pre-OCR preparation to a subprocess path, and spilling prepared OCR images to temporary PNG files while keeping lightweight metadata rows in the parent process.
- Exposed `--no-temp-ocr-image-files` on `convert`, `smoke`, and `doctor workflow`, keeping disk-backed prepared OCR images as the default retained path and adding focused CLI regression coverage for the opt-out wiring.
- Documented the retained temporary OCR image file behavior in `README.md` and `DISCLAIMER.md`, including normal cleanup, residue risk on forced termination, and the `--no-temp-ocr-image-files` opt-out.

## [0.3.3] - 2026-04-16
- Hardened `convert` and `smoke` artifact handling by rejecting path collisions, applying overwrite protection to sidecar outputs, and requiring explicit `--input-sup` for `smoke`.
- Hardened local setup and runtime safety by restricting `setup --force` to managed cache targets and moving llama-server manager state into a user-private runtime root with advisory metadata only.
- Fixed `doctor runtime paddle` so Paddle profile overrides are normalized correctly and runtime checks no longer crash while rendering results.
- Switched SRT, detector/corrector manifests, Gemini cache, and Gemini auth-config writes to shared atomic file writers.
- Refined the Gemini corrector with bounded concurrency, cache-miss coordination, tuned retry defaults, and per-row fallback to baseline text when request retries are exhausted.

## [0.3.2] - 2026-04-14
- Added post-parse exact-image deduplication in the pipeline for baseline OCR, `ocr-fast`, detector reuse, local Qwen reuse, and Gemini correction reuse.
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
