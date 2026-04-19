# Changelog

## [0.3.5] - 2026-04-19
- Pinned the built-in setup model bundles to explicit upstream revisions and verified their downloaded artifacts against repository-maintained SHA-256 hashes, while keeping custom setup values available as user-managed paths.
- Kept managed `llama-server` traffic on a dedicated internal request host so wildcard bind settings no longer force health checks and OCR requests onto the exposed bind address.
- Switched the project to the release `gguf==0.18.0` package, accepted installed `gguf` packages directly for mmproj work, and kept the pinned snapshot only as an explicit fallback source.
- Removed auto-created `smoke` and `doctor workflow` temporary artifact directories after successful runs, while keeping failed runs and explicit smoke output directories available for inspection.
- Aligned the root CLI `smoke` help text with the current explicit `--input-sup` requirement so the top-level help no longer advertises a bundled sample default.
- Rejected invalid detector-related `smoke` flag combinations with smoke-specific errors so wrapper validation no longer leaks the hidden convert-only `--detector-output` flag.
- Clarified the README onboarding flow so `uv sync` remains the dependency-install step and `istots setup` is described as runtime-asset provisioning only.

## [0.3.4] - 2026-04-18
- Reduced memory pressure, improved safety for local OCR runs on constrained systems, and gave users clearer control over temporary OCR image handling.

## [0.3.3] - 2026-04-16
- Blocked avoidable output mistakes in `convert` and `smoke`, and required more explicit smoke inputs.
- Hardened local setup and runtime management so routine maintenance is less likely to touch files or processes outside the intended workspace.
- Fixed Paddle runtime diagnosis so doctor checks return structured results instead of crashing mid-report.
- Hardened generated outputs and local cache state so interrupted writes are less likely to leave broken files behind.
- Improved Gemini correction throughput and kept conversions moving when the API is slow or unstable.

## [0.3.2] - 2026-04-14
- Reduced repeated OCR work across identical subtitle images so conversions can finish with less wasted compute.
- Reduced a runaway main OCR failure mode where the model could repeat irrelevant text until token limits were exhausted, so bad recognitions are less likely to overwhelm the result.

## [0.3.1] - 2026-04-14
- Improved subtitle extraction on complex multi-window PGS streams so more real-world layouts are reconstructed correctly before OCR.
- Reduced the cost of furigana masking so batches with heavy furigana can run faster without changing the intended masking behavior.

## [0.3.0] - 2026-04-13
- Shifted the product toward a stronger local OCR workflow so higher-quality conversion is available by default, with fallback kept as an explicit backup path.
- Simplified local OCR asset preparation so users can reach a working retained runtime with less manual setup work.
- Added faster review and correction paths so users can balance speed, disagreement checking, and conservative cleanup based on subtitle quality needs.
- Added setup and validation tools so users can confirm the environment is ready before trusting full conversion runs.

## [0.2.0] - 2026-04-05
- Improved subtitle readability by optionally suppressing furigana before OCR when it would otherwise pollute the text output.
- Expanded the converter to handle multi-window subtitle layouts and give users more control over how overlapping cues are written.

## [0.1.0] - 2026-04-05
- Initial release
