# Changelog

## [0.3.0] - 2026-04-12
- Step 1: introduced an OCR engine boundary with shared backend contracts and a factory that separates engine selection from the conversion pipeline while keeping the current HF runtime behavior intact.
- Generalized the product-facing compute surface from `cuda/cpu` to `gpu/cpu`, with backend-specific device mapping handled internally.
- Added regression coverage for OCR engine selection, backend initialization fallback, device mapping, and CLI help output.
- Step 2: expanded `istots setup` to provision the retained HF fallback assets, GGUF runtime assets, and a derived `min_pixels=32768` GGUF mmproj.
- Added pinned gguf snapshot support and a deterministic GGUF mmproj materializer aligned with the retained experiment contract.
- Added regression coverage for GGUF asset download, pinned gguf loading, mmproj materialization, and the expanded setup CLI surface.
- Step 3: added a product-owned `llama-server` runtime-management layer with retained launch profiles, per-role asset resolution, and launch-command construction.
- Added `istots doctor --engine llama-server` with retained readiness checks for binary discovery, asset presence, port readiness, launch readiness, and minimal smoke requests.
- Added reusable explicit llama OpenAI sampling defaults and regression coverage for runtime management and doctor flows.
- Step 4: switched the default `convert` OCR path to the retained `llama-server` runtime while keeping `HF` as an explicit fallback engine.
- Added a product-owned `llama-server` OCR backend that validates retained runtime assets, launches the retained OCR role, serves OCR requests, and shuts the runtime down cleanly after conversion.
- Exposed retained `convert` runtime overrides for profile, port, threads, GPU layers, mmproj offload, and startup timeout, and added regression coverage for CLI routing, backend construction, and OCR backend lifecycle.
- Step 5: added the retained optional faster OCR mode for `convert`, keeping retained `default` as the primary OCR path.
- Added retained hybrid branch routing with non-tall rows sent to `ocr-fast`, tall rows sent to retained `ocr`, batch-by-branch execution, and original-order restoration before SRT assembly.
- Added regression coverage for fast-mode CLI validation, hybrid OCR branch partitioning, retained role selection, and row-order restoration.

## [0.2.0] - 2026-04-05
- Added optional furigana masking before OCR to reduce furigana noise in generated subtitles.
- Implemented a multi-window PGS subtitle pipeline with window-aware OCR, debug/export tooling, GUI inspection, and selectable `safe`/`overlap` SRT output policies.

## [0.1.0] - 2026-04-05
- Initial release
