# Changelog

## [0.3.0] - 2026-04-12
- Step 1: introduced an OCR engine boundary with shared backend contracts and a factory that separates engine selection from the conversion pipeline while keeping the current HF runtime behavior intact.
- Generalized the product-facing compute surface from `cuda/cpu` to `gpu/cpu`, with backend-specific device mapping handled internally.
- Added regression coverage for OCR engine selection, backend initialization fallback, device mapping, and CLI help output.
- Step 2: expanded `istots setup` to provision the retained HF fallback assets, GGUF runtime assets, and a derived `min_pixels=32768` GGUF mmproj.
- Added pinned gguf snapshot support and a deterministic GGUF mmproj materializer aligned with the retained experiment contract.
- Added regression coverage for GGUF asset download, pinned gguf loading, mmproj materialization, and the expanded setup CLI surface.

## [0.2.0] - 2026-04-05
- Added optional furigana masking before OCR to reduce furigana noise in generated subtitles.
- Implemented a multi-window PGS subtitle pipeline with window-aware OCR, debug/export tooling, GUI inspection, and selectable `safe`/`overlap` SRT output policies.

## [0.1.0] - 2026-04-05
- Initial release
