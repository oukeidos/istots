# Task

## Session Goal

Implement the experimentally validated feature set into the product code without regressions, using `../experiments_260409` as the source of truth and `IMPLEMENTATION_PLAN.md` as the execution reference.

## Practical Implementation Order Status

- Step 1. Engine Boundary First: completed
- Step 2. Setup / Materialization Contract: completed
- Step 3. Runtime Management Contract: completed
- Step 4. Primary OCR Convert Flow: completed
- Step 5. Optional Faster OCR Path: completed
- Step 6. Default Detector Integration: completed
- Step 7. Conservative Correction Integration: pending
- Step 8. `HF` Optional Fallback Wiring: pending
- Step 9. Minimal Regression Probe Wiring: pending
- Step 10. Final Documentation Pass: pending

## Current Focus

- Step 1 completed: introduced a stable OCR engine boundary that separates engine selection and backend construction from the conversion pipeline while preserving current HF behavior.
- The product-facing compute surface now uses `gpu` / `cpu` terminology, with backend-specific mapping handled internally.
- Step 2 completed: `setup` now prepares the retained HF fallback assets, GGUF runtime assets, and the derived `min_pixels=32768` GGUF mmproj using the pinned gguf contract.
- Step 3 completed: added product-owned `llama-server` runtime profiles, retained-role asset resolution, launch-command construction, and a doctor/preflight surface with readiness smoke checks.
- Step 4 completed: switched the baseline convert path to the retained `llama-server` OCR runtime while preserving the HF path as an explicit fallback engine and exposing retained runtime overrides on the CLI.
- Step 5 completed: added the retained optional faster OCR mode with non-tall `ocr-fast` routing, tall retained `ocr` routing, batch-by-branch execution, and original-order restoration before SRT assembly.
- Step 6 completed: added retained hybrid detector manifest generation with non-tall `ocr-fast` alternate-read comparisons, tall `detector` repeat-drift comparisons, and disagreement labeling for correction-ready downstream use.
- Next focus: Step 7 conservative correction integration on top of the retained hybrid detector output.
