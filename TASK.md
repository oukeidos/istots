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
- Post-plan credential/setup expansion completed: `setup` can now opt into retained local Qwen corrector provisioning with the default `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` asset, and `auth gemini` now manages Gemini credentials through keyring-first storage with a configured `.env` fallback path.
- Post-plan OCR execution cleanup completed: the product no longer exposes batch execution, and OCR now runs sequentially one subtitle image at a time across convert, smoke, detector, and corrector paths.
- Post-plan mmproj offload cleanup completed: runtime profiles no longer force `--no-mmproj-offload`, and local Qwen correction now exposes it only as an explicit opt-in override.
- Post-plan runtime profile cleanup completed: removed the no-op `memory` runtime profile and kept only `auto` and `cpu` as supported product profiles.
- Post-plan hardware policy cleanup completed: split engine-specific hardware control so `llama-server` now relies only on runtime profiles while HF uses dedicated `hf-device` and `hf-dtype` controls.
- Post-plan Qwen recipe cleanup completed: removed local Qwen thread-count hardcodes, kept only behavior-critical retained fields such as context size and reasoning mode, and stopped steering `llama-server` through PyTorch CUDA checks.
- Post-plan HF fast-path parity completed: `--engine hf --ocr-mode fast` now uses the retained hybrid branch rule with default HF reading on tall rows and retained `min_pixels=32768` only on non-tall rows.

## Completed Design Work

- Implemented model-family structured `llama-server` overrides so PaddleOCR-VL runtime launches and Qwen3.5 corrector runtime launches can receive different low-level launch settings in the same `convert` or `smoke` execution.
- Scope boundary:
  - keep shared runtime infrastructure such as `llama-server` binary discovery and shared host selection separate from model-family override policy
  - treat PaddleOCR-VL runtime roles (`ocr`, `ocr-fast`, `detector`) as one model-family override surface
  - treat the Qwen3.5 corrector runtime (`corrector`) as a separate model-family override surface
  - keep role-local differences such as role names, default ports, and mmproj selection as internal resolution details rather than user-facing tuning families
- User-facing outcome:
  - removed the remaining misleading global low-level `llama-server` override surface from `convert` and `smoke`
  - replaced it with explicit structured override families for PaddleOCR-VL and Qwen3.5
  - preserved `doctor` as a single-role inspection tool with direct role-level override input
- Proposed internal contract:
  - introduce product-owned structured override objects for `PaddleOCRVLRuntimeOverrides` and `Qwen35RuntimeOverrides`
  - resolve those family-level objects into final launch specs through internal role-local defaults
  - let `ocr`, `ocr-fast`, and `detector` inherit the same PaddleOCR-VL tuning inputs while preserving their internal role-local asset and port defaults
  - let `corrector` resolve from the Qwen3.5 tuning inputs without inheriting Paddle-specific runtime policy
- Acceptance criteria status:
  - one `convert` run can pass different override sets to PaddleOCR-VL and Qwen3.5 runtimes without cross-contamination
  - `--qwen-no-mmproj-offload` and the wider Qwen family override surface only affect the Qwen corrector launch
  - PaddleOCR-VL launch behavior remains unchanged when only Qwen overrides are changed
  - regression tests lock the resolved launch spec for mixed Paddle/Qwen runs and verify that internal role defaults remain stable

## Planned Design Work: Detector Expansion

- Retained experiment read:
  - keep the current hybrid detector as the default detector surface:
    - non-tall:
      - alternate-read disagreement against the retained `min_pixels=32768` branch
    - tall:
      - repeat-drift disagreement against a repeated retained default read
  - add an opt-in wider detector surface:
    - default detector surface plus a wider default-repeat detector slice
    - practical meaning:
      - one extra same-default repeated read that is evaluated as an additional detector pass
  - keep dominant-family widening as a separate optional recall add-on:
    - separate family add-on layer
    - not part of the default detector
    - not equivalent to the wider detector extension
- Product posture to preserve:
  - detector default remains the current hybrid surface
  - wider detector remains opt-in only
  - dominant-family add-on remains opt-in and explicitly recall-oriented
  - detector stays `llama-server`-only in the product surface
- Proposed execution shape:
  - baseline OCR pass:
    - retained default OCR output
  - detector pass 1:
    - current hybrid detector pass
    - non-tall:
      - `ocr-fast`
    - tall:
      - `detector`
  - detector pass 2:
    - repeated retained default read for the wider detector extension
    - produce wider default-repeat detector rows only when the wider detector mode is enabled
  - add-on layer:
    - dominant-family candidate enrichment runs after detector rows are assembled
    - may be attached to:
      - the default detector surface
      - the wider detector surface
    - but remains logically separate from detector-pass generation
- Internal contract:
  - replace the current single detector-record builder with a detector-surface builder that can compose:
    - the default detector surface
    - the wider detector surface
    - optional dominant-family add-on
  - preserve per-row provenance in manifests:
    - `detector_branch`
    - `alternate_source_kind`
    - surface membership tags such as:
      - `hybrid_detector`
      - `p2_meaningful_temp0`
      - `dominant_family_addon`
  - keep correction trigger selection surface-aligned:
    - default correction uses the default detector surface
    - opt-in wider correction may consume the wider detector surface
    - family add-on rows are available only when explicitly requested
- User-facing outcome:
  - add a detector mode surface with at least:
    - default hybrid detector
    - wider detector
  - add an explicit family add-on toggle separate from detector mode
  - do not overload the default detector flag into silently enabling either wider mode or family enrichment
- Acceptance criteria:
  - default detector output remains backward-compatible with the current default detector surface
  - wider detector adds only the retained wider default-repeat slice on top of the default detector surface
  - family add-on can be attached independently to either the default detector surface or the wider detector surface
  - manifests preserve enough provenance to reconstruct the default detector surface, the wider detector surface, and add-on-exclusive slices
  - regression tests lock row-surface membership and prevent default-surface expansion by accident
- Current implementation status:
  - completed the first retained slice:
    - default detector surface plus dominant-family add-on
  - the add-on is now opt-in on top of the default detector surface through `--detector-family-addon`
  - dominant-family extraction is currently limited to repeated single-character kanji families seen in the live detector disagreements
  - selection now uses row-level `support`, `pure`, `mixed`, and agreement-breadth gating instead of a count-only winner
  - add-on `alternate_text` is synthesized as the paired family-character swap, matching the retained `family_pair_swap` contract
  - wider detector mode is now available through `--detector-mode wider`
