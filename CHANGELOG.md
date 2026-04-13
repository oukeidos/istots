# Changelog

## [0.3.1] - 2026-04-13
- Ported the retained PGS parser refinement by restoring validated parser/full-surface semantics, sharing frame assembly across parser and `sup_reader`, adding the OCR-facing compose-bypass path, and verifying the result with targeted and sample-backed regression suites.
- Ported the retained furigana-mask optimization by adding same-SUP duplicate analysis reuse, retained process-based batch parallelism, histogram-backed thresholding, row-run connected components, dense-array union-find bookkeeping, and slab/band line construction without changing the validated output contract.
- Revalidated the furigana-mask port against `tests/test_furigana_mask.py` and `tests/test_pipeline_batch.py`, matching the retained handoff expectation of `35 passed`.
- Confirmed the furigana-mask port on the shared whole-SUP validator corpus with `mismatch_count=0` across stage 1 and stage 2 and runtime remaining in the retained post-optimization regime.

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
