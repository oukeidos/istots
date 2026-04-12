# IStoTS: Image Subtitles to Text Subtitles

`istots` converts Blu-ray `SUP` subtitles into `SRT`. The main stack is
PaddleOCR-VL 1.5 GGUF through `llama-server`, with an optional Hugging Face OCR
fallback, an optional local Qwen3.5 correction route, and an optional cloud
Gemini correction route.

Beyond plain OCR, two features matter most. First, `istots` can detect likely
error groups and send only those rows to a stronger, more expensive correction
model instead of rewriting the whole subtitle stream. Second, it can apply a
furigana masking pass before OCR to suppress small ruby-like side annotations
that often become noise in downstream text workflows such as translation or
other subtitle reformatting.

## Setup

Start by cloning the repository and entering the project directory:

```bash
git clone https://github.com/oukeidos/istots.git
cd istots
```

You then need Python 3.11 or newer and `uv`. For the primary OCR path, you
also need a working `llama-server` binary on the host system. `istots` does
not install `llama-server` for you. It looks for the binary from
`--llama-server-path`, `ISTOTS_LLAMA_SERVER_PATH`, `PATH`, and a fallback
local path in that order. For `llama-server` installation, follow the official
`llama.cpp` documentation. The project README covers the main install paths,
including prebuilt binaries and source builds, and the wiki includes platform
notes:

- <https://github.com/ggml-org/llama.cpp>
- <https://github.com/ggml-org/llama.cpp/wiki>

With the repository checked out, install the Python dependencies:

```bash
uv sync
```

Then prepare the default local runtime assets:

```bash
uv run istots setup
```

This installs the core Python dependencies and prepares the default local
runtime assets. The setup command downloads the retained PaddleOCR-VL GGUF
model, the base mmproj, the derived `min_pixels=32768` mmproj used by the
fast OCR branch, and the local files needed for the optional Hugging Face
fallback.

If you want to run actual OCR inference through `--engine hf`, install the HF
runtime dependencies as well:

```bash
uv sync --extra hf
```

The HF route does not depend on `llama.cpp` or `llama-server` for OCR
execution. It is useful as a fallback when you want a pure HF runtime, but it
is usually much slower than the primary `llama-server` path. It also does not
support the full feature surface: detector and correction features remain tied
to the `llama-server` route.

If you want `istots` to provision the default local Qwen corrector assets, run
setup with:

```bash
uv run istots setup --with-qwen-corrector
```

The current default local Qwen corrector assets are
`unsloth/Qwen3.5-35B-A3B-GGUF`, with
`Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` as the model file and
`mmproj-BF16.gguf` as the default mmproj file.

Gemini API key setup is handled separately through `istots auth gemini`. A
typical first step for the cloud corrector is:

```bash
uv run istots auth gemini set
```

## Basic Use

The simplest conversion command is:

```bash
uv run istots input.sup output.srt
```

This runs the default PaddleOCR-VL `llama-server` path, reads subtitle images
from `input.sup`, and writes a plain `SRT` file to `output.srt`. By default,
the run does not enable detector manifests, correction, or furigana masking.

If you want the fallback HF OCR route instead, run:

```bash
uv run istots input.sup output.srt --engine hf
```

## Core Features

### OCR

`llama-server` is the default OCR route. `--engine hf` remains in the project
mainly as a fallback for users who do not want to install or manage the
`llama.cpp` runtime. It is useful for accessibility and easier setup on some
systems, but it is not the preferred route. In the retained personal
experiment notes, `llama-server` was consistently faster, with roughly a `2x`
to `3x` advantage over HF on GPU and about a `4x` to `5x` advantage over HF
`float32` on CPU. For practical CPU-only use, `llama-server` is therefore
strongly recommended. The HF route also remains a reduced fallback surface:
detector and correction features stay on the `llama-server` path.

The fast OCR mode works by changing the image budget, not the prompt. The key
knob is `min_pixels`. On the `llama-server` route, the wide-row fast branch
uses a derived `min_pixels=32768` mmproj. On the HF route, it uses a
processor-side `min_pixels=32768` override. Tall rows stay on the default path
because the lower `min_pixels` setting is much less reliable there. In the
retained experiment notes, the adopted hybrid branch rule was about `1.35x`
faster than the all-default path while staying close to parity in the reviewed
slice. These figures come from personal experiment notes, not from a broad
benchmark.

```bash
uv run istots input.sup output.srt --ocr-mode fast
uv run istots input.sup output.srt --engine hf --ocr-mode fast
```

### Automatic Error Correction

The correction workflow is deliberately narrow. Instead of sending the whole
subtitle stream to a stronger model, `istots` repeats OCR on the same base
model and uses the disagreements as a detector. In practice, this means
rerunning the same OCR path under slightly different conditions, then sending
only the differing rows to a stronger but more expensive local or cloud model
for re-OCR. The final merge is conservative and tries to replace only the
ambiguous segment instead of rewriting the whole row.

The detector exists because repeated OCR reads are not perfectly stable.
Under `llama-server`, repeating the default read or switching to the lower
`min_pixels` read can produce drift. In small-sample tests, that drift was not
mostly random. Many rows kept falling into the same small set of alternate
readings, often just a two-way split. On one 100-row sample, `22 / 100` rows
drifted under repeated `temp=0.0` reads, but only `5 / 100` were meaningfully
different, and most of the drifted rows collapsed into simple binary variants.
The same pattern appeared again on a larger 238-row drift-focused set, where
most drifting rows still collapsed into recurring binary variants. That is the
core idea behind the detector: treat recurring disagreement as a signal that a
row is uncertain enough to justify escalation. In small-scale tests, this
worked well enough to be useful rather than theoretical. On a reviewed default
correction set, the baseline output was accepted on `66.2%` of rows, while the
same rows rose to `91.5%` with local Qwen3.5-35B-A3B and `100%` with Gemini 3.1 Pro Preview.
These figures come from small reviewed slices, not from a broad benchmark.

In normal use, you enable a corrector and let `istots` handle the detector
stage automatically. The local corrector path uses a retained Qwen3.5 recipe
through `llama-server`. The current default local model is
`Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` from
`unsloth/Qwen3.5-35B-A3B-GGUF`:

```bash
uv run istots input.sup output.srt --corrector qwen-local
```

If the default Qwen assets are not installed, you can point at explicit local
files:

```bash
uv run istots input.sup output.srt \
  --corrector qwen-local \
  --corrector-model-path /path/to/qwen.gguf \
  --corrector-mmproj-path /path/to/qwen-mmproj.gguf
```

The Gemini path uses the same detector-triggered structure, but the actual
correction request goes to the Gemini API. The current default Gemini model is
`gemini-3.1-pro-preview`:

```bash
uv run istots auth gemini set
uv run istots input.sup output.srt --corrector gemini
```

The detector has two main scopes. The default scope is narrower and is the
recommended everyday setting. The wider scope adds one more repeat-read
surface, so it catches more uncertain rows at the cost of more correction
work. In the reviewed notes, widening the detector increased the reviewed row
count from `71` to `87`, or about `1.23x`. On those added rows, both Qwen and
Gemini still performed well, so the wider mode is useful when recall matters
more than cost.

There is also an optional dominant-family add-on. This is a kanji-specific
feature, not a general text rule. It looks for a repeated kanji confusion
family and then expands the detector to include more rows from that family. It
was designed that way because the idea works best when the script has a large
character inventory, as kanji does. If the character set is much smaller, the
same strategy would likely push too many weak candidates into the detector and
produce noisy over-expansion. Even with kanji, the trade-off is much steeper.
In the reviewed notes, turning the add-on on expanded the reviewed row count
from `87` to `205`, or about `2.36x`, so the correction workload grows much
faster than the extra recall. That is why it remains an optional recall-heavy
mode rather than part of the default detector.

```bash
uv run istots input.sup output.srt \
  --detector-mode wider \
  --detector-family-addon
```

The local Qwen route uses a comparatively large model, so it may require a
host with enough VRAM or RAM and, on some systems, host-specific runtime
settings. If local correction fails to start cleanly, check the Qwen runtime
options before assuming the correction path itself is broken.

### Furigana Masking

Japanese Blu-ray subtitles often contain small side annotations that are
useful for a viewer but can become noise once the subtitle text is reused in
other workflows such as translation, editing, or format conversion.
`--furigana-mask` runs an image heuristic before OCR and tries to suppress
those regions. Concretely, it binarizes the subtitle image, extracts connected
components, estimates whether the text flow is mainly horizontal or vertical,
groups components into line-like clusters, and then looks for thinner
side-aligned fragments that resemble furigana rather than main subtitle text.
Those selected regions are masked before OCR. It is an image heuristic, not a
linguistic parser, so it is not guaranteed to help on every file. The
practical way to use it is comparative: run the same subtitle set with and
without masking and keep the result that is more useful for the downstream
text workflow you care about.

```bash
uv run istots input.sup output.srt --furigana-mask
```

## Validation and Support Tools

`doctor` is structured around the actual product surfaces. `doctor runtime
paddle` checks the retained PaddleOCR-VL runtime family. `doctor runtime qwen`
checks the local Qwen corrector runtime family. `doctor auth gemini` checks
Gemini API key availability without printing the key. `doctor workflow ...` runs a
real workflow smoke on an input SUP file that you specify explicitly.

```bash
uv run istots doctor runtime paddle
uv run istots doctor runtime qwen
uv run istots doctor auth gemini
uv run istots doctor workflow default --input-sup /path/to/input.sup
uv run istots doctor workflow corrector-gemini --input-sup /path/to/input.sup
```

`smoke` is a convenience wrapper over the same retained product surfaces used
by `convert`. It writes an SRT and, when relevant, detector and correction
manifests into a temporary directory or a directory you choose.

```bash
uv run istots smoke --input-sup /path/to/input.sup
uv run istots smoke --input-sup /path/to/input.sup --ocr-mode fast
uv run istots smoke --input-sup /path/to/input.sup --corrector qwen-local
```

Gemini API keys are managed through `auth gemini`. The recommended default is
keyring-backed storage, because the key is kept out of shell history and out
of project files. The simplest setup is:

```bash
uv run istots auth gemini set
uv run istots auth gemini status
```

If you prefer an `.env` file, create one with the standard key name:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Then point `istots` at that file:

```bash
uv run istots auth gemini env-file set /path/to/.env
uv run istots auth gemini status
```

This is useful when you already manage local secrets through `.env` files or
when keyring is unavailable on the host.

You can also provide the key through the current shell environment instead of
storing it through `istots`. For a one-off shell session:

```bash
export GEMINI_API_KEY=your_api_key_here
uv run istots input.sup output.srt --corrector gemini
```

For a more persistent shell setup on `bash`, add the same export line to
`~/.bashrc` or `~/.profile`, then reload the shell:

```bash
echo 'export GEMINI_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

If you do not want to use the default variable name, `--corrector-api-key-env`
lets you point `istots` at a different environment variable.

Gemini API key resolution for `--corrector gemini` currently follows this
order: local keyring first, configured `.env` file second, and current shell
environment last.

## Command Reference

This section is intentionally later in the document. The earlier sections
explain the product model first. For exact syntax, `uv run istots --help`
remains the authoritative CLI reference.

### `convert`

`convert` is the main command. The positional arguments are `input.sup` and
`output.srt`. The most important switches are `--engine` for backend choice,
`--ocr-mode` for default versus fast OCR scheduling, `--furigana-mask` for
pre-OCR masking, `--detector-output` and `--detector-mode` for disagreement
inspection, `--detector-family-addon` for the recall-oriented kanji-family
surface, `--corrector` and `--corrector-output` for conservative correction,
and `--srt-policy`, `--max-items`, `--max-new-tokens`, `--force`, and
`--quiet` for execution control.

The `llama-server` path exposes model-family runtime overrides rather than a
single global low-level surface. PaddleOCR-VL tuning uses `--paddle-profile`,
`--paddle-port`, `--paddle-threads`, `--paddle-threads-batch`,
`--paddle-gpu-layers`, `--paddle-no-mmproj-offload`, and
`--paddle-startup-timeout-sec`. Qwen corrector tuning uses `--qwen-profile`,
`--qwen-port`, `--qwen-threads`, `--qwen-threads-batch`,
`--qwen-gpu-layers`, `--qwen-no-mmproj-offload`, `--qwen-ctx-size`,
`--qwen-n-predict`, `--qwen-reasoning`, and
`--qwen-startup-timeout-sec`. Shared runtime infrastructure still uses
`--llama-server-path`.

The HF path has its own explicit hardware controls. `--hf-device` selects
`auto`, `cpu`, or `gpu`. `--hf-dtype` selects `auto`, `float32`, `float16`,
or `bfloat16`. `--model-id` selects the HF model or a local HF model path.

### `setup`

`setup` prepares local model artifacts. In common use, the important flags are
`--with-qwen-corrector` when you also want the default local Qwen assets,
`--model-id` for the HF fallback model, `--gguf-model-id` for the primary
Paddle GGUF repository, and `--models-dir` or `--support-dir` when you want
custom local storage locations. The more technical flags
`--gguf-py-base-url`, `--gguf-source-mode`, `--min-pixels`, and `--force`
exist for reproducibility and explicit materialization control.

### `smoke`

`smoke` is a thin convenience layer over the same retained product surfaces
used by `convert`. It accepts `--input-sup`, `--output-dir`, `--ocr-mode`,
`--no-detector`, `--detector-mode`, `--detector-family-addon`,
`--corrector`, `--furigana-mask`, `--srt-policy`, `--force`, and `--quiet`,
plus the same Paddle and Qwen runtime override families used by `convert`.

### `doctor`

`doctor` uses a structured two-part form. The runtime targets are `paddle` and
`qwen`. The auth target is `gemini`. The workflow targets are `default`,
`wider`, `corrector-qwen`, and `corrector-gemini`, and all workflow checks
require `--input-sup`. Structured doctor runs also accept the relevant Paddle
or Qwen model-family runtime overrides and `--api-key-env` for Gemini auth and
workflow checks.

### `auth`

`auth gemini` manages Gemini API keys through `set`, `delete`, `status`,
`env-file set`, and `env-file clear`. The command is intentionally narrow: it
stores or clears the API key, reports whether a usable key source exists, and
does not print the key itself.

### `materialize-mmproj`

`materialize-mmproj` exists for direct control of the derived Paddle mmproj
artifact. Most users do not need it because `istots setup` already runs the
materializer. The command takes a positional `base_mmproj` path and uses
`--output`, `--min-pixels`, `--support-dir`, `--gguf-py-base-url`,
`--gguf-source-mode`, `--force`, and `--quiet` for explicit low-level control.

## Environment Variables

The most relevant environment variables are `ISTOTS_LLAMA_SERVER_PATH` for the
`llama-server` binary, `ISTOTS_MODELS_DIR` for the local model cache root,
`ISTOTS_SUPPORT_DIR` for the pinned `gguf` support cache,
`ISTOTS_AUTH_CONFIG_PATH` for the local Gemini auth config file, and
`GEMINI_API_KEY` for shell-based Gemini API key resolution.

## Language Support

Language support should be read in two layers. The main OCR engine and the
correction models do not serve the same role. PaddleOCR-VL is the primary OCR
engine and is responsible for the first-pass subtitle read. Qwen3.5 and
Gemini are used only as narrow correction models on detector-selected rows, so
their language coverage matters as multilingual correction capability rather
than as the primary OCR guarantee.

For the main OCR engine, PaddleOCR documents PaddleOCR-VL as supporting 109
languages. Its published list covers a broad mix of Latin, Cyrillic, Arabic,
Devanagari, and East Asian languages, including English, Chinese, Korean,
Japanese, Thai, Arabic, Hindi, French, German, Spanish, Portuguese, Russian,
and many others.

The local Qwen3.5 corrector belongs to the multilingual Qwen3.5 family, which
Qwen describes as supporting 201 languages and dialects. In this project,
that broad coverage is useful as multilingual correction support, not as a
replacement for the primary OCR engine.

Gemini should be read the same way. Google documents Gemini as trained to work
with 38 languages, including English, Chinese, Korean, Japanese, Arabic,
Hindi, French, German, Spanish, Portuguese, Russian, Thai, Turkish,
Ukrainian, Vietnamese, and others. Here too, the role is correction on
detector-selected rows, not the first OCR pass.

Official references:

- PaddleOCR-VL language list: <https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html>
- Qwen3.5 multilingual support: <https://qwen.ai/blog?id=qwen3.5>
- Gemini model language support: <https://ai.google.dev/gemini-api/docs/models>
