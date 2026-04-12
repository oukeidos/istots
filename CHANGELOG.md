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
- Step 6: added retained hybrid detector manifest generation on `convert` for the default disagreement surface.
- Added non-tall alternate-read detector routing through `ocr-fast`, tall repeat-drift routing through the retained `detector` role, and disagreement labeling for correction-ready downstream use.
- Added regression coverage for detector CLI validation, detector manifest generation, and detector text-difference assessment.
- Step 7: added opt-in conservative correction on `convert` using the retained hybrid detector as the default disagreement trigger surface.
- Ported the retained anchor-only merge logic into product code, added local Qwen correction wiring with the retained strict runtime recipe, and added Gemini tall-row prompt gating.
- Added correction manifest export and regression coverage for anchor merge, correction CLI validation, and detector-to-correction pipeline composition.
- Step 8: moved the heavyweight HF runtime behind an explicit optional dependency contract while keeping `hf` as the explicit fallback engine.
- Updated HF-facing CLI and runtime messaging to point users to the optional HF install path and added packaging regression coverage for the retained fallback dependency posture.
- Step 9: added a dedicated `istots smoke` quick-validation command that defaults to the retained `../test/sample.sup` asset.
- Auto-wired smoke SRT, detector, and correction artifact paths for the retained primary `llama-server` workflow and kept fast-mode smoke available on the same sample.
- Added regression coverage for the new smoke CLI surface and for the frozen `sample.sup` parser contract used by required smoke tests.
- Step 10: refreshed the product-facing README around the retained setup boundary, runtime profiles, host patterns, and the default smoke workflow.
- Documented the retained detector/corrector posture and the recommended quick-validation order from setup through doctor, smoke, and full convert.
- Post-plan stability fix: changed the OCR backend lifecycle so heavy OCR runtimes are opened one role at a time instead of being kept resident concurrently across fast, detector, and local-corrector stages.
- Added regression coverage to lock single-backend residency across retained hybrid OCR, detector, and correction flows.
- Post-plan corrector fix: wrapped Gemini image inputs in the required `inline_data` part envelope so product Gemini correction requests match the retained experiment contract.
- Added regression coverage for Gemini inline-image request construction and verified retained Qwen local and Gemini correction smoke flows on experiment-backed SUP input slices.
- Post-plan runtime-manager fix: moved `llama-server` launch ownership behind a single cross-role manager that serializes runtime residency, records managed process state, cleans stale managed runtimes, and rejects unexpected reserved-port conflicts before launch.
- Added regression coverage for managed stale-runtime cleanup, managed state teardown, and reserved-port conflict detection, and verified the real Qwen local correction slice leaves no `llama-server` process behind after completion.
- Post-plan credential/setup expansion: added opt-in `setup --with-qwen-corrector` provisioning for the retained default Qwen corrector asset pair from `unsloth/Qwen3.5-35B-A3B-GGUF`.
- Added `auth gemini` key management with hidden-input keyring storage, configured `.env` fallback paths, keyring-first credential resolution, and regression coverage for the new setup/auth surfaces.
- Post-plan OCR execution cleanup: removed the product batch-execution surface and now run OCR sequentially per subtitle image across convert, smoke, detector, and corrector flows.
- Removed the `--batch-size` CLI surface, simplified the OCR pipeline and backend execution path, and added regression coverage for the sequential-only behavior.
- Post-plan mmproj offload cleanup: stopped forcing `--no-mmproj-offload` through shared runtime profiles and removed the default force from local Qwen correction.
- Added an opt-in `--qwen-no-mmproj-offload` override for `qwen-local` and locked the new launch behavior with runtime, CLI, and pipeline regression coverage.
- Post-plan runtime profile cleanup: removed the no-op `memory` profile from the product surface and kept `auto` and `cpu` as the supported runtime profiles.
- Post-plan hardware policy cleanup: removed the legacy global device surface from `llama-server` flows and split hardware control into `llama-server` runtime profiles plus HF-only `--hf-device` / `--hf-dtype` options.
- Stopped using PyTorch CUDA detection to steer `llama-server`, limited auto CPU fallback to the HF engine, and removed local Qwen thread-count hardcodes while keeping only behavior-critical recipe fields.
- Post-plan model-family runtime override cleanup: replaced the last global low-level `llama-server` override surface on `convert` and `smoke` with separate PaddleOCR-VL and Qwen3.5 override families.
- Added regression coverage to lock model-family override routing, shared Paddle role resolution, and isolated Qwen local runtime overrides.
- Post-plan HF fast-path parity: added retained hybrid HF fast OCR so non-tall rows use `min_pixels=32768` while tall rows stay on the default HF read.
- Added regression coverage for HF processor `min_pixels` override routing and for the mixed tall/wide HF fast branch contract.
- Post-plan text-diff decoupling: split generic text normalization from the retained Japanese OCR orthographic rules without changing the product default detector behavior.
- Added regression coverage to lock the retained Japanese default profile and the new internal generic profile behavior separately.
- Post-plan detector add-on expansion: added an opt-in dominant-family recall layer that can attach directly to the retained `S1` detector without enabling the wider `S2` detector path.
- Restricted the family add-on to repeated single-character kanji families, synthesized add-on `alternate_text` through family-pair swap instead of an extra OCR read, and added regression coverage for add-on manifest generation, kanji-only family inference, and CLI routing.

## [0.2.0] - 2026-04-05
- Added optional furigana masking before OCR to reduce furigana noise in generated subtitles.
- Implemented a multi-window PGS subtitle pipeline with window-aware OCR, debug/export tooling, GUI inspection, and selectable `safe`/`overlap` SRT output policies.

## [0.1.0] - 2026-04-05
- Initial release
