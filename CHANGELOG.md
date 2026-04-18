# Changelog

## [0.3.4] - 2026-04-18
- Reduced memory pressure and made local OCR runs easier to operate safely on constrained systems, while giving users clearer control over temporary OCR image handling.

## [0.3.3] - 2026-04-16
- Made `convert` and `smoke` safer to run by blocking avoidable output mistakes and requiring more explicit smoke inputs.
- Made local setup and runtime management safer so routine maintenance is less likely to touch files or processes outside the intended workspace.
- Made Paddle runtime diagnosis easier to trust by ensuring doctor checks return structured results instead of crashing mid-report.
- Made generated outputs and local cache state more resilient so interrupted writes are less likely to leave broken files behind.
- Made Gemini correction runs more usable in practice by improving throughput and keeping conversions moving when the API is slow or unstable.

## [0.3.2] - 2026-04-14
- Reduced repeated OCR work across identical subtitle images so conversions can finish with less wasted compute.
- Reduced a runaway main OCR failure mode where the model could repeat irrelevant text until token limits were exhausted, so bad recognitions are less likely to overwhelm the result.

## [0.3.1] - 2026-04-14
- Improved subtitle extraction on complex multi-window PGS streams so more real-world layouts are reconstructed correctly before OCR.
- Reduced the cost of furigana masking so batches with heavy furigana can run faster without changing the intended masking behavior.

## [0.3.0] - 2026-04-13
- Shifted the product toward a stronger local OCR workflow so higher-quality conversion is available by default, with fallback kept as an explicit backup path.
- Made the required local OCR assets easier to prepare so users can reach a working retained runtime with less manual setup work.
- Added faster review and correction paths so users can balance speed, disagreement checking, and conservative cleanup based on subtitle quality needs.
- Added setup and validation tools so users can confirm the environment is ready before trusting full conversion runs.

## [0.2.0] - 2026-04-05
- Improved subtitle readability by optionally suppressing furigana before OCR when it would otherwise pollute the text output.
- Expanded the converter to handle multi-window subtitle layouts and give users more control over how overlapping cues are written.

## [0.1.0] - 2026-04-05
- Initial release
