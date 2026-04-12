# Task

## Session Goal

Implement the experimentally validated feature set into the product code without regressions, using `../experiments_260409` as the source of truth and `IMPLEMENTATION_PLAN.md` as the execution reference.

## Practical Implementation Order Status

- Step 1. Engine Boundary First: completed
- Step 2. Setup / Materialization Contract: completed
- Step 3. Runtime Management Contract: pending
- Step 4. Primary OCR Convert Flow: pending
- Step 5. Optional Faster OCR Path: pending
- Step 6. Default Detector Integration: pending
- Step 7. Conservative Correction Integration: pending
- Step 8. `HF` Optional Fallback Wiring: pending
- Step 9. Minimal Regression Probe Wiring: pending
- Step 10. Final Documentation Pass: pending

## Current Focus

- Step 1 completed: introduced a stable OCR engine boundary that separates engine selection and backend construction from the conversion pipeline while preserving current HF behavior.
- The product-facing compute surface now uses `gpu` / `cpu` terminology, with backend-specific mapping handled internally.
- Step 2 completed: `setup` now prepares the retained HF fallback assets, GGUF runtime assets, and the derived `min_pixels=32768` GGUF mmproj using the pinned gguf contract.
- Next focus: Step 3 runtime-management contract, including product-owned runtime profiles and doctor/preflight readiness checks.
