# DISCLAIMER

`istots` is a subtitle conversion tool. It reads Blu-ray `SUP` subtitle images,
runs OCR on them, and writes `SRT` text subtitles.

By using this project, you accept the following.

This document is a practical notice only. It does not replace `LICENSE`,
`THIRD_PARTY_NOTICES.md`, or the terms of any third-party software, model, or
service used with this project.

## 1) Intended Use

`istots` is designed for assisted subtitle extraction and conversion.

- Input: image-based subtitle data from `SUP`
- Processing: subtitle frame parsing, deduplication, OCR
- Output: text-based subtitle data in `SRT`

The output is a working result, not a guaranteed final subtitle master.
Manual review is expected.

## 2) No Warranty

The software is provided "as is".

There is no warranty of:

- correctness
- completeness
- fitness for a particular purpose
- availability
- performance
- non-infringement

You are responsible for validating both the software and its output before any
real-world use.

## 3) Limitation of Liability

To the maximum extent permitted by law, the authors and contributors are not
liable for any direct, indirect, incidental, special, consequential, or other
damages arising from use of the software, the generated subtitles, or related
artifacts.

## 4) OCR Accuracy and Output Quality

OCR can be wrong, incomplete, or unstable.

Examples:

- characters may be misread
- line breaks may be wrong
- timestamps may be imperfect
- repeated frames may be merged in ways you do not want
- empty or very short subtitle events may be skipped
- stylized text, low-contrast text, vertical text, or noisy subtitle images may
  degrade output quality

This project also does not perform full editorial cleanup.
For example, furigana removal is not part of the main conversion path.

Do not use unreviewed output in contexts where subtitle errors can cause legal,
financial, safety, compliance, or other material harm.

## 5) Models, Dependencies, and Third-Party Terms

`istots` depends on third-party software and model artifacts, including the
Python runtime stack and the OCR model selected for the project.

In the default setup, this includes:

- Hugging Face model hosting for model download during setup
- PyTorch and Transformers for inference
- the configured OCR model, currently `PaddlePaddle/PaddleOCR-VL-1.5`

Third-party licenses, terms, and restrictions apply to those components.
The project license does not replace them.

See:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- upstream model and package terms

## 6) Network and Local Data Handling

The normal `istots convert` path is intended to use a locally available model.
However, `istots setup` downloads model files from a third-party host.

Local execution may create or use:

- cached model files
- input `SUP` files
- generated `SRT` files
- exported subtitle images
- temporary OCR image files in the OS temporary directory
- logs and temporary runtime data

You are responsible for choosing safe storage locations, access controls, and
backup policies for that data.

If the subtitle content is sensitive, private, or not yours to redistribute,
handle the input and output files accordingly.

On the default local OCR path, these temporary OCR image files are removed when
the workflow finishes normally. If the process is killed or the system crashes,
they can remain in the temporary directory. If your local policy does not allow
that behavior, use:

- `--no-temp-ocr-image-files`

## 7) Rights and Legal Compliance

You must have the legal right to process the subtitle source material and to
use, store, modify, or distribute any resulting subtitle output.

You are responsible for compliance with applicable law and policy, including:

- copyright
- contract and platform terms
- privacy and data protection rules
- workplace, institutional, or customer confidentiality requirements

Using this tool does not grant rights to the original subtitle stream, the
underlying video content, or any generated derivative work.

## 8) Operational Limits and Failure Modes

The project may fail because of:

- missing model files
- incompatible package versions
- unsupported hardware
- insufficient CPU, GPU, RAM, or VRAM
- malformed or unexpected `SUP` input
- OCR model load failures
- runtime errors in third-party libraries

Even when execution succeeds, the output can still be wrong.
Test on a small sample before large or production-style runs.

## 9) Security Boundaries

This project reduces some risk by using a local model path during conversion,
but it does not eliminate supply-chain or environment risk.

You still assume risk for:

- compromised upstream packages or model artifacts
- poisoned local caches
- unsafe local environment configuration
- insecure handling of generated subtitle files

Treat dependency updates, model changes, and execution environment changes as
security-relevant events.

## 10) No Affiliation

This project is not affiliated with, endorsed by, or sponsored by PaddlePaddle,
Hugging Face, PyTorch, or any other third-party provider referenced by the
project.

Names of third-party products and organizations are used only to identify the
relevant software or model artifacts.

## 11) User Responsibility

You are responsible for:

- setup and environment configuration
- reviewing subtitle output
- verifying timing and text accuracy
- checking legal and policy compliance
- deciding whether the tool is suitable for your workflow

If you use this project, you accept these limits and responsibilities.
