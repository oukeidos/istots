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

An optional desktop GUI is included as a foundation for a future
Windows-focused packaged distribution. If you want to try it from source, use
`uv sync --extra gui` and then launch it with `uv run istots-gui`.

Then prepare the default local runtime assets:

```bash
uv run istots setup
```

This prepares the default local runtime assets after `uv sync` has installed
the core Python dependencies. The default setup command downloads the retained
PaddleOCR-VL GGUF model, the base mmproj, and the derived `min_pixels=32768`
mmproj used by the fast OCR branch. For the built-in bundles that `setup`
provisions, `istots setup` pins explicit upstream revisions and verifies the
downloaded artifacts against repository-maintained SHA-256 hashes.

If you want to run actual OCR inference through `--engine hf`, install the HF
runtime dependencies as well:

```bash
uv sync --extra hf
```

Then provision the retained local HF fallback bundle explicitly:

```bash
uv run istots setup --with-hf-fallback
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

If you override the HF fallback model id with `--with-hf-fallback --model-id`,
the GGUF model id, or the default Qwen filenames, `istots` still allows that
custom setup path, but revision pinning and artifact hash verification become
user-managed for that bundle.

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

### Temporary OCR Image Files

To keep RAM use lower, the default local OCR path may write temporary OCR image
files to the OS temporary directory during conversion. These files are normal
subtitle image crops, not the full source video. On a normal run, `istots`
removes them when the workflow finishes. If the process is killed or the system
crashes, they can remain in the temporary directory.

If your local policy does not allow temporary OCR image files on disk, disable
that path and keep the OCR images in memory instead:

```bash
uv run istots input.sup output.srt --no-temp-ocr-image-files
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

By default, both `smoke` and `doctor workflow ...` use an internal temporary
artifact directory and remove it after a successful run. If a run fails, the
temporary directory is retained for inspection. If you pass `smoke --output-dir`,
that directory is treated as user-managed and is left in place.

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

- The lists below cover the public command surface shown by `--help`.
- Hidden compatibility aliases are intentionally omitted here.
- Every command also accepts `-h`, `--help`.

### `convert`

- Purpose:
  - Main SUP-to-SRT conversion command.
- Positional arguments:
  - `input_sup`: Input `.sup` file to read.
  - `output_srt`: Output `.srt` file to write.
- Flags:
  - `--engine`: Select the OCR backend, `llama-server` or `hf`.
  - `--hf-device`: Select the HF execution device when `--engine hf` is used.
  - `--hf-dtype`: Select the HF tensor dtype when `--engine hf` is used.
  - `--model-id`: Choose the HF model id or local HF model path.
  - `--models-dir`: Override the local model cache root.
  - `--max-items`: Limit processing to the first `N` subtitle rows.
  - `--max-new-tokens`: Set the OCR token budget per image.
  - `--ocr-mode`: Choose the retained `default` or `fast` OCR path.
  - `--paddle-profile`: Select the PaddleOCR-VL `llama-server` runtime profile.
  - `--llama-server-path`: Point at an explicit `llama-server` binary.
  - `--paddle-port`: Override the PaddleOCR-VL `llama-server` port.
  - `--paddle-threads`: Override PaddleOCR-VL thread count.
  - `--paddle-threads-batch`: Override PaddleOCR-VL batch thread count.
  - `--paddle-gpu-layers`: Override PaddleOCR-VL GPU layer count.
  - `--paddle-no-mmproj-offload`: Disable mmproj offload for PaddleOCR-VL.
  - `--paddle-startup-timeout-sec`: Set the PaddleOCR-VL startup timeout.
  - `--quiet`: Suppress progress logs.
  - `--furigana-mask`: Enable the pre-OCR furigana masking pass.
  - `--no-temp-ocr-image-files`: Keep prepared OCR images only in memory instead of writing temporary OCR image files.
  - `--detector-output`: Write detector disagreements to a JSONL manifest.
  - `--detector-mode`: Choose the retained `default` or `wider` detector surface.
  - `--detector-family-addon`: Enable the dominant-family detector add-on.
  - `--corrector`: Choose `off`, `qwen-local`, or `gemini` correction.
  - `--corrector-output`: Write correction records to a JSONL file.
  - `--corrector-model-path`: Point at an explicit local Qwen GGUF model.
  - `--corrector-mmproj-path`: Point at an explicit local Qwen mmproj.
  - `--qwen-profile`: Select the Qwen `llama-server` runtime profile.
  - `--qwen-port`: Override the Qwen `llama-server` port.
  - `--qwen-threads`: Override Qwen thread count.
  - `--qwen-threads-batch`: Override Qwen batch thread count.
  - `--qwen-gpu-layers`: Override Qwen GPU layer count.
  - `--qwen-no-mmproj-offload`: Disable mmproj offload for Qwen.
  - `--qwen-ctx-size`: Override Qwen context size.
  - `--qwen-n-predict`: Override Qwen output token count.
  - `--qwen-reasoning`: Override Qwen reasoning mode.
  - `--qwen-startup-timeout-sec`: Set the Qwen startup timeout.
  - `--corrector-gemini-model`: Choose the Gemini model id.
  - `--corrector-api-key-env`: Choose the environment variable used for the Gemini API key.
  - `--corrector-thinking-level`: Set the Gemini thinking level.
  - `--corrector-media-resolution`: Set the Gemini media resolution hint.
  - `--corrector-cache-dir`: Choose the Gemini request cache directory.
  - `--corrector-gemini-max-attempts`: Set the Gemini retry-attempt limit.
  - `--corrector-gemini-request-timeout-sec`: Set the Gemini per-request timeout.
  - `--corrector-gemini-max-workers`: Set the in-process Gemini parallelism limit.
  - `--srt-policy`: Choose `safe` or `overlap` SRT assembly.
  - `--force`: Overwrite existing output artifacts if they already exist.

### `setup`

- Purpose:
  - Download and materialize the primary local OCR assets, plus optional HF fallback and corrector assets.
- Flags:
  - `--with-hf-fallback`: Also provision the retained HF fallback model bundle.
  - `--model-id`: Choose the HF fallback model id to download when `--with-hf-fallback` is enabled.
  - `--gguf-model-id`: Choose the PaddleOCR-VL GGUF repository to download.
  - `--with-qwen-corrector`: Also provision the retained local Qwen corrector assets.
  - `--qwen-corrector-model-id`: Choose the Qwen corrector repository id.
  - `--qwen-corrector-model-filename`: Choose the Qwen GGUF model filename.
  - `--qwen-corrector-mmproj-filename`: Choose the Qwen mmproj filename.
  - `--models-dir`: Override the local model cache root.
  - `--force`: Re-download and re-materialize cached assets.
  - `--support-dir`: Override the local support cache root.
  - `--gguf-py-base-url`: Override the source root for the optional pinned `gguf-py` snapshot fallback.
  - `--gguf-source-mode`: Choose how the `gguf` implementation is sourced.
  - `--min-pixels`: Set the derived `mmproj` `min_pixels` value.
  - `--quiet`: Suppress progress logs.

### `smoke`

- Purpose:
  - Run a quick validation workflow over the retained product surface.
- Flags:
  - `--input-sup`: Required. Choose the SUP file used for smoke validation.
  - `--output-dir`: Choose the directory for smoke artifacts. Without this flag,
    smoke uses a temporary directory and removes it after a successful run.
  - `--models-dir`: Override the local model cache root.
  - `--max-new-tokens`: Set the OCR token budget per image.
  - `--ocr-mode`: Choose the retained `default` or `fast` OCR path.
  - `--paddle-profile`: Select the PaddleOCR-VL runtime profile.
  - `--llama-server-path`: Point at an explicit `llama-server` binary.
  - `--paddle-port`: Override the PaddleOCR-VL `llama-server` port.
  - `--paddle-threads`: Override PaddleOCR-VL thread count.
  - `--paddle-threads-batch`: Override PaddleOCR-VL batch thread count.
  - `--paddle-gpu-layers`: Override PaddleOCR-VL GPU layer count.
  - `--paddle-no-mmproj-offload`: Disable mmproj offload for PaddleOCR-VL smoke runs.
  - `--paddle-startup-timeout-sec`: Set the PaddleOCR-VL startup timeout.
  - `--quiet`: Suppress progress logs.
  - `--furigana-mask`: Enable the pre-OCR furigana masking pass.
  - `--no-temp-ocr-image-files`: Keep prepared OCR images only in memory instead of writing temporary OCR image files.
  - `--no-detector`: Skip detector manifest generation during smoke validation.
  - `--detector-mode`: Choose the retained `default` or `wider` detector surface.
  - `--detector-family-addon`: Enable the dominant-family detector add-on.
  - `--corrector`: Choose `off`, `qwen-local`, or `gemini` correction.
  - `--corrector-model-path`: Point at an explicit local Qwen GGUF model.
  - `--corrector-mmproj-path`: Point at an explicit local Qwen mmproj.
  - `--qwen-profile`: Select the Qwen `llama-server` runtime profile.
  - `--qwen-port`: Override the Qwen `llama-server` port.
  - `--qwen-threads`: Override Qwen thread count.
  - `--qwen-threads-batch`: Override Qwen batch thread count.
  - `--qwen-gpu-layers`: Override Qwen GPU layer count.
  - `--qwen-no-mmproj-offload`: Disable mmproj offload for Qwen.
  - `--qwen-ctx-size`: Override Qwen context size.
  - `--qwen-n-predict`: Override Qwen output token count.
  - `--qwen-reasoning`: Override Qwen reasoning mode.
  - `--qwen-startup-timeout-sec`: Set the Qwen startup timeout.
  - `--corrector-gemini-model`: Choose the Gemini model id.
  - `--corrector-api-key-env`: Choose the environment variable used for the Gemini API key.
  - `--corrector-thinking-level`: Set the Gemini thinking level.
  - `--corrector-media-resolution`: Set the Gemini media resolution hint.
  - `--corrector-cache-dir`: Choose the Gemini request cache directory.
  - `--corrector-gemini-max-attempts`: Set the Gemini retry-attempt limit.
  - `--corrector-gemini-request-timeout-sec`: Set the Gemini per-request timeout.
  - `--corrector-gemini-max-workers`: Set the in-process Gemini parallelism limit.
  - `--srt-policy`: Choose `safe` or `overlap` SRT assembly.
  - `--force`: Overwrite any existing smoke artifacts.

### `doctor`

- Purpose:
  - Run structured runtime, auth, and workflow diagnostics.
- Positional arguments:
  - `doctor_category`: Choose `runtime`, `auth`, or `workflow`.
  - `doctor_target`: Choose the target within that category.
- Flags:
  - `--models-dir`: Override the local model cache root.
  - `--min-pixels`: Set the fast-role asset `min_pixels` value used by doctor.
  - `--llama-server-path`: Point at an explicit `llama-server` binary.
  - `--host`: Choose the host to bind or probe.
  - `--input-sup`: Provide the SUP file required by workflow checks.
  - `--api-key-env`: Choose the environment variable used for Gemini auth checks.
  - `--paddle-profile`: Select the PaddleOCR-VL runtime profile.
  - `--paddle-port`: Override the PaddleOCR-VL `llama-server` port.
  - `--paddle-threads`: Override PaddleOCR-VL thread count.
  - `--paddle-threads-batch`: Override PaddleOCR-VL batch thread count.
  - `--paddle-gpu-layers`: Override PaddleOCR-VL GPU layer count.
  - `--paddle-no-mmproj-offload`: Disable mmproj offload for PaddleOCR-VL doctor runs.
  - `--paddle-startup-timeout-sec`: Set the PaddleOCR-VL startup timeout.
  - `--no-temp-ocr-image-files`: Keep prepared OCR images only in memory instead of writing temporary OCR image files during `doctor workflow ...`.
  - `--corrector-model-path`: Point at an explicit local Qwen GGUF model.
  - `--corrector-mmproj-path`: Point at an explicit local Qwen mmproj.
  - `--qwen-profile`: Select the Qwen runtime profile.
  - `--qwen-port`: Override the Qwen `llama-server` port.
  - `--qwen-threads`: Override Qwen thread count.
  - `--qwen-threads-batch`: Override Qwen batch thread count.
  - `--qwen-gpu-layers`: Override Qwen GPU layer count.
  - `--qwen-no-mmproj-offload`: Disable mmproj offload for Qwen doctor runs.
  - `--qwen-ctx-size`: Override Qwen context size.
  - `--qwen-n-predict`: Override Qwen output token count.
  - `--qwen-reasoning`: Override Qwen reasoning mode.
  - `--qwen-startup-timeout-sec`: Set the Qwen startup timeout.
  - `--quiet`: Suppress progress logs.

For `doctor workflow ...`, the command keeps its auto-created temporary
artifacts only when the workflow check fails. Successful workflow doctor runs
remove the temporary artifact directory after reporting the check result.

### `auth`

- Purpose:
  - Manage Gemini credential sources.
- Positional command structure:
  - `gemini`: Enter the Gemini credential namespace.
  - `set`: Store the Gemini API key in the local keyring.
  - `delete`: Delete the Gemini API key from the local keyring.
  - `status`: Report whether usable Gemini credentials are available.
  - `env-file set path`: Store the fallback Gemini `.env` file path.
  - `env-file clear`: Clear the configured fallback Gemini `.env` file path.
- Flags:
  - No command-specific flags beyond `-h`, `--help`.

### `materialize-mmproj`

- Purpose:
  - Build a derived Paddle `mmproj` artifact directly from a base `mmproj`.
- Positional arguments:
  - `base_mmproj`: Path to the official base `mmproj` GGUF file.
- Flags:
  - `--output`: Choose the derived `mmproj` output path.
  - `--min-pixels`: Set the derived `mmproj` `min_pixels` value.
  - `--support-dir`: Override the local support cache root.
  - `--gguf-py-base-url`: Override the source root for the optional pinned `gguf-py` snapshot fallback.
  - `--gguf-source-mode`: Choose how the `gguf` implementation is sourced.
  - `--force`: Overwrite an existing derived `mmproj`.
  - `--quiet`: Suppress progress logs.

## Environment Variables

The most relevant environment variables are `ISTOTS_LLAMA_SERVER_PATH` for the
`llama-server` binary, `ISTOTS_MODELS_DIR` for the local model cache root,
`ISTOTS_SUPPORT_DIR` for the optional pinned `gguf` snapshot support cache,
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
