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
- Step 7. Conservative Correction Integration: completed
- Step 8. `HF` Optional Fallback Wiring: completed
- Step 9. Minimal Regression Probe Wiring: completed
- Step 10. Final Documentation Pass: completed

## Current Focus

- Step 1 completed: introduced a stable OCR engine boundary that separates engine selection and backend construction from the conversion pipeline while preserving current HF behavior.
- The product-facing compute surface now uses `gpu` / `cpu` terminology, with backend-specific mapping handled internally.
- Step 2 completed: `setup` now prepares the retained HF fallback assets, GGUF runtime assets, and the derived `min_pixels=32768` GGUF mmproj using the pinned gguf contract.
- Step 3 completed: added product-owned `llama-server` runtime profiles, retained-role asset resolution, launch-command construction, and a doctor/preflight surface with readiness smoke checks.
- Step 4 completed: switched the baseline convert path to the retained `llama-server` OCR runtime while preserving the HF path as an explicit fallback engine and exposing retained runtime overrides on the CLI.
- Step 5 completed: added the retained optional faster OCR mode with non-tall `ocr-fast` routing, tall retained `ocr` routing, batch-by-branch execution, and original-order restoration before SRT assembly.
- Step 6 completed: added retained hybrid detector manifest generation with non-tall `ocr-fast` alternate-read comparisons, tall `detector` repeat-drift comparisons, and disagreement labeling for correction-ready downstream use.
- Step 7 completed: added opt-in conservative correction on `convert`, retained anchor-only merge behavior, local Qwen correction wiring, and Gemini tall-row prompt gating on top of the retained hybrid detector trigger surface.
- Step 8 completed: moved the heavyweight HF runtime behind an explicit optional dependency contract while keeping `hf` as a simple explicit fallback engine.
- Step 9 completed: added a dedicated `smoke` quick-validation command around the retained `../test/sample.sup` asset, auto-wired retained smoke artifacts, and locked the sample parser contract with regression coverage.
- Step 10 completed: refreshed the product-facing README around setup boundaries, runtime profiles, host patterns, the retained smoke workflow, and the detector/corrector posture.
- The Practical Implementation Order is complete for this session.
- Post-plan stability fix completed: the OCR pipeline now opens heavy OCR runtimes sequentially, one role at a time, to avoid concurrent residency during hybrid OCR, detector, and local-corrector flows.
- Post-plan corrector fix completed: Gemini correction requests now send image parts with the required `inline_data` envelope, and experiment-backed Qwen local / Gemini correction smoke runs both completed successfully on a detector-positive slice.
- Post-plan runtime-manager fix completed: `llama-server` launches now flow through a single manager that serializes runtime ownership across OCR, detector, and corrector roles, persists managed state for stale-runtime cleanup, and blocks unexpected reserved-port conflicts before launch.
