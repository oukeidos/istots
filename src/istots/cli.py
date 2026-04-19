from __future__ import annotations

import argparse
import getpass
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from istots import __version__
from istots.corrector import (
    DEFAULT_GEMINI_MAX_ATTEMPTS,
    DEFAULT_GEMINI_MAX_WORKERS,
    DEFAULT_GEMINI_REQUEST_TIMEOUT_SEC,
)
from istots.model_store import (
    DEFAULT_GGUF_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME,
    DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME,
    DEFAULT_QWEN_CORRECTOR_MODEL_ID,
)
from istots.ocr import LOCAL_PADDLE_CTX_SIZE, PaddleOCRVLRuntimeOverrides, Qwen35RuntimeOverrides

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


class _RootHelpParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._subcommand_parsers: list[argparse.ArgumentParser] = []

    def register_subcommand_parser(self, parser: argparse.ArgumentParser) -> None:
        self._subcommand_parsers.append(parser)

    def format_help(self) -> str:
        help_text = super().format_help().rstrip()
        if not self._subcommand_parsers:
            return f"{help_text}\n"

        sections = [help_text, "", "Subcommand Details:"]
        for parser in self._subcommand_parsers:
            sections.extend(["", parser.format_help().rstrip()])
        return "\n".join(sections) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = _RootHelpParser(
        prog="istots",
        description="Convert SUP subtitles to SRT using PaddleOCR-VL (offline by default).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    convert = subparsers.add_parser(
        "convert",
        help="Convert SUP to SRT (offline; uses locally downloaded model)",
    )
    parser.register_subcommand_parser(convert)
    _add_convert_arguments(convert)

    setup = subparsers.add_parser(
        "setup",
        help="Download model artifacts from Hugging Face to local cache",
    )
    parser.register_subcommand_parser(setup)
    _add_setup_arguments(setup)

    materialize_mmproj = subparsers.add_parser(
        "materialize-mmproj",
        help="Create a min_pixels-tuned llama.cpp mmproj GGUF from an official base mmproj",
    )
    parser.register_subcommand_parser(materialize_mmproj)
    _add_materialize_mmproj_arguments(materialize_mmproj)

    doctor = subparsers.add_parser(
        "doctor",
        help="Run structured runtime, auth, and workflow doctor checks",
    )
    parser.register_subcommand_parser(doctor)
    _add_doctor_arguments(doctor)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run quick validation on an explicit input SUP",
    )
    parser.register_subcommand_parser(smoke)
    _add_smoke_arguments(smoke)

    auth = subparsers.add_parser(
        "auth",
        help="Manage Gemini API credentials and fallback configuration",
    )
    parser.register_subcommand_parser(auth)
    _add_auth_arguments(auth)

    return parser


def _build_convert_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="istots convert")
    _add_convert_arguments(parser)
    return parser


def _build_smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="istots smoke")
    _add_smoke_arguments(parser)
    return parser


def _build_doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="istots doctor")
    _add_doctor_arguments(parser)
    return parser


def _add_temp_ocr_image_file_argument(parser: argparse.ArgumentParser, *, help_suffix: str = "") -> None:
    suffix = f" {help_suffix}" if help_suffix else ""
    parser.add_argument(
        "--no-temp-ocr-image-files",
        action="store_true",
        help=(
            "Keep prepared OCR images only in memory instead of writing temporary OCR image files."
            " This avoids temporary OCR image files on disk but uses more RAM."
            f"{suffix}"
        ),
    )


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_sup", type=Path, help="Input .sup file")
    parser.add_argument("output_srt", type=Path, help="Output .srt file")
    parser.add_argument(
        "--engine",
        choices=("llama-server", "hf"),
        default="llama-server",
        help="OCR engine selection (default: llama-server). `hf` is the explicit optional fallback path.",
    )
    parser.add_argument(
        "--hf-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="HF-only device selection when using `--engine hf` (default: auto)",
    )
    parser.add_argument(
        "--hf-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="HF-only torch dtype policy when using `--engine hf` (default: auto)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "HF model ID or local HF model path for `--engine hf`. "
            "If a model ID is given, it must already exist in local cache from `istots setup`. "
            "The HF engine also requires the optional HF runtime."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Local model cache root (default: ~/.cache/istots/models "
            "or ISTOTS_MODELS_DIR)."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Process only first N subtitle items (debugging)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per subtitle image",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=("default", "fast"),
        default="default",
        help=(
            "OCR mode for `convert`: retained default OCR or the optional fast hybrid path. "
            "On llama-server, fast uses `ocr-fast` for non-tall rows; on HF, fast uses "
            "retained `min_pixels=32768` only for non-tall rows. (default: default)"
        ),
    )
    parser.add_argument(
        "--paddle-profile",
        choices=("auto", "cpu"),
        default="auto",
        help="PaddleOCR-VL llama-server runtime profile when using `--engine llama-server` (default: auto)",
    )
    parser.add_argument("--runtime-profile", dest="paddle_profile", choices=("auto", "cpu"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--llama-server-path",
        type=Path,
        default=None,
        help="Explicit llama-server binary path for `--engine llama-server`",
    )
    parser.add_argument(
        "--paddle-port",
        type=int,
        default=None,
        help="Override the shared PaddleOCR-VL llama-server port for convert",
    )
    parser.add_argument("--runtime-port", dest="paddle_port", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-threads",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server thread count",
    )
    parser.add_argument("--threads", dest="paddle_threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-threads-batch",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server batch thread count",
    )
    parser.add_argument("--threads-batch", dest="paddle_threads_batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-gpu-layers",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server GPU layer count",
    )
    parser.add_argument("--gpu-layers", dest="paddle_gpu_layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-no-mmproj-offload",
        action="store_true",
        help="Disable mmproj offload for PaddleOCR-VL llama-server runs",
    )
    parser.add_argument("--no-mmproj-offload", dest="paddle_no_mmproj_offload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-startup-timeout-sec",
        type=float,
        default=120.0,
        help="PaddleOCR-VL llama-server startup timeout in seconds",
    )
    parser.add_argument("--startup-timeout-sec", dest="paddle_startup_timeout_sec", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-ctx-size",
        type=int,
        default=None,
        help=f"Override PaddleOCR-VL llama-server context size (default policy: {LOCAL_PADDLE_CTX_SIZE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )
    parser.add_argument(
        "--furigana-mask",
        action="store_true",
        help="Enable optional furigana masking before OCR (default: disabled)",
    )
    _add_temp_ocr_image_file_argument(parser)
    parser.add_argument(
        "--detector-output",
        type=Path,
        default=None,
        help=(
            "Write retained hybrid detector disagreements to a JSONL manifest. "
            "Requires `--engine llama-server` with `--ocr-mode default`."
        ),
    )
    parser.add_argument(
        "--detector-mode",
        choices=("default", "wider"),
        default="default",
        help=(
            "Detector surface selection for retained llama-server detector flows. "
            "`wider` adds the retained wider default-repeat detector slice on top of the default detector surface."
        ),
    )
    parser.add_argument(
        "--detector-family-addon",
        action="store_true",
        help=(
            "Opt into the retained dominant-family agreement-row add-on on top of the "
            "default detector surface. The add-on only considers repeated single-char kanji families."
        ),
    )
    parser.add_argument(
        "--corrector",
        choices=("off", "qwen-local", "gemini"),
        default="off",
        help=(
            "Attach the retained conservative corrector to convert. "
            "Requires `--engine llama-server` with `--ocr-mode default`."
        ),
    )
    parser.add_argument(
        "--corrector-output",
        type=Path,
        default=None,
        help="Optional JSONL path for retained conservative correction records.",
    )
    parser.add_argument(
        "--corrector-model-path",
        type=Path,
        default=None,
        help="Explicit local GGUF corrector model path for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--corrector-mmproj-path",
        type=Path,
        default=None,
        help="Explicit local GGUF corrector mmproj path for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-profile",
        choices=("auto", "cpu"),
        default="auto",
        help="Qwen3.5 llama-server runtime profile for `--corrector qwen-local`",
    )
    parser.add_argument(
        "--qwen-port",
        type=int,
        default=None,
        help="Override the Qwen3.5 llama-server port for `--corrector qwen-local`.",
    )
    parser.add_argument("--corrector-port", dest="qwen_port", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--qwen-threads",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server thread count for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-threads-batch",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server batch thread count for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-gpu-layers",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server GPU layer count for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-no-mmproj-offload",
        action="store_true",
        help="Force `--no-mmproj-offload` for `--corrector qwen-local`.",
    )
    parser.add_argument("--corrector-no-mmproj-offload", dest="qwen_no_mmproj_offload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--qwen-ctx-size",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server context size for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-n-predict",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server `-n` value for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-reasoning",
        default=None,
        help="Override Qwen3.5 llama-server reasoning mode for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-startup-timeout-sec",
        type=float,
        default=120.0,
        help="Qwen3.5 llama-server startup timeout in seconds for `--corrector qwen-local`.",
    )
    parser.add_argument("--corrector-startup-timeout-sec", dest="qwen_startup_timeout_sec", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--corrector-gemini-model",
        default="gemini-3.1-pro-preview",
        help="Gemini model id for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name holding the Gemini API key.",
    )
    parser.add_argument(
        "--corrector-thinking-level",
        default="low",
        help="Optional Gemini thinking level for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-media-resolution",
        default=None,
        help="Optional Gemini media resolution level for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for `--corrector gemini` requests.",
    )
    parser.add_argument(
        "--corrector-gemini-max-attempts",
        type=int,
        default=DEFAULT_GEMINI_MAX_ATTEMPTS,
        help="Maximum Gemini retry attempts for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-gemini-request-timeout-sec",
        type=float,
        default=DEFAULT_GEMINI_REQUEST_TIMEOUT_SEC,
        help="Per-request timeout in seconds for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-gemini-max-workers",
        type=int,
        default=DEFAULT_GEMINI_MAX_WORKERS,
        help="Maximum in-process parallel Gemini requests for `--corrector gemini`.",
    )
    parser.add_argument(
        "--srt-policy",
        choices=("safe", "overlap"),
        default="safe",
        help="SRT output policy: merge simultaneous windows safely or keep overlapping cues",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output artifacts without prompting",
    )


def _add_smoke_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-sup",
        type=Path,
        default=None,
        help="Quick-validation SUP path (required)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for smoke artifacts. Without this flag, smoke uses a temporary "
            "directory and removes it after a successful run."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Local model cache root (default: ~/.cache/istots/models "
            "or ISTOTS_MODELS_DIR)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per subtitle image",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=("default", "fast"),
        default="default",
        help=(
            "Quick-validation OCR mode for the retained primary engine "
            "(default: default)"
        ),
    )
    parser.add_argument(
        "--paddle-profile",
        choices=("auto", "cpu"),
        default="auto",
        help="PaddleOCR-VL llama-server runtime profile for smoke validation (default: auto)",
    )
    parser.add_argument("--runtime-profile", dest="paddle_profile", choices=("auto", "cpu"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--llama-server-path",
        type=Path,
        default=None,
        help="Explicit llama-server binary path for smoke validation",
    )
    parser.add_argument(
        "--paddle-port",
        type=int,
        default=None,
        help="Override the shared PaddleOCR-VL llama-server port for smoke validation",
    )
    parser.add_argument("--runtime-port", dest="paddle_port", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-threads",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server thread count for smoke validation",
    )
    parser.add_argument("--threads", dest="paddle_threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-threads-batch",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server batch thread count for smoke validation",
    )
    parser.add_argument("--threads-batch", dest="paddle_threads_batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-gpu-layers",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server GPU layer count for smoke validation",
    )
    parser.add_argument("--gpu-layers", dest="paddle_gpu_layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-no-mmproj-offload",
        action="store_true",
        help="Disable mmproj offload for PaddleOCR-VL smoke validation",
    )
    parser.add_argument("--no-mmproj-offload", dest="paddle_no_mmproj_offload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-startup-timeout-sec",
        type=float,
        default=120.0,
        help="PaddleOCR-VL llama-server startup timeout in seconds for smoke validation",
    )
    parser.add_argument("--startup-timeout-sec", dest="paddle_startup_timeout_sec", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--paddle-ctx-size",
        type=int,
        default=None,
        help=f"Override PaddleOCR-VL llama-server context size for smoke validation (default policy: {LOCAL_PADDLE_CTX_SIZE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )
    parser.add_argument(
        "--furigana-mask",
        action="store_true",
        help="Enable optional furigana masking before OCR (default: disabled)",
    )
    _add_temp_ocr_image_file_argument(parser, help_suffix="Applies to smoke conversion.")
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="Skip the retained hybrid detector manifest in smoke validation",
    )
    parser.add_argument(
        "--detector-mode",
        choices=("default", "wider"),
        default="default",
        help=(
            "Detector surface selection for smoke validation. "
            "`wider` adds the retained wider default-repeat detector slice on top of the default detector surface."
        ),
    )
    parser.add_argument(
        "--detector-family-addon",
        action="store_true",
        help=(
            "Opt into the retained dominant-family agreement-row add-on on top of the "
            "default smoke detector surface. The add-on only considers repeated single-char kanji families."
        ),
    )
    parser.add_argument(
        "--corrector",
        choices=("off", "qwen-local", "gemini"),
        default="off",
        help=(
            "Attach the retained conservative corrector to smoke validation. "
            "Requires `--ocr-mode default`."
        ),
    )
    parser.add_argument(
        "--corrector-model-path",
        type=Path,
        default=None,
        help="Explicit local GGUF corrector model path for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--corrector-mmproj-path",
        type=Path,
        default=None,
        help="Explicit local GGUF corrector mmproj path for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-profile",
        choices=("auto", "cpu"),
        default="auto",
        help="Qwen3.5 llama-server runtime profile for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-port",
        type=int,
        default=None,
        help="Override the Qwen3.5 llama-server port for `--corrector qwen-local`.",
    )
    parser.add_argument("--corrector-port", dest="qwen_port", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--qwen-threads",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server thread count for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-threads-batch",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server batch thread count for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-gpu-layers",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server GPU layer count for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-no-mmproj-offload",
        action="store_true",
        help="Force `--no-mmproj-offload` for `--corrector qwen-local`.",
    )
    parser.add_argument("--corrector-no-mmproj-offload", dest="qwen_no_mmproj_offload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--qwen-ctx-size",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server context size for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-n-predict",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server `-n` value for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-reasoning",
        default=None,
        help="Override Qwen3.5 llama-server reasoning mode for `--corrector qwen-local`.",
    )
    parser.add_argument(
        "--qwen-startup-timeout-sec",
        type=float,
        default=120.0,
        help="Qwen3.5 llama-server startup timeout in seconds for `--corrector qwen-local`.",
    )
    parser.add_argument("--corrector-startup-timeout-sec", dest="qwen_startup_timeout_sec", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--corrector-gemini-model",
        default="gemini-3.1-pro-preview",
        help="Gemini model id for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name holding the Gemini API key.",
    )
    parser.add_argument(
        "--corrector-thinking-level",
        default="low",
        help="Optional Gemini thinking level for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-media-resolution",
        default=None,
        help="Optional Gemini media resolution level for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for `--corrector gemini` requests.",
    )
    parser.add_argument(
        "--corrector-gemini-max-attempts",
        type=int,
        default=DEFAULT_GEMINI_MAX_ATTEMPTS,
        help="Maximum Gemini retry attempts for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-gemini-request-timeout-sec",
        type=float,
        default=DEFAULT_GEMINI_REQUEST_TIMEOUT_SEC,
        help="Per-request timeout in seconds for `--corrector gemini`.",
    )
    parser.add_argument(
        "--corrector-gemini-max-workers",
        type=int,
        default=DEFAULT_GEMINI_MAX_WORKERS,
        help="Maximum in-process parallel Gemini requests for `--corrector gemini`.",
    )
    parser.add_argument(
        "--srt-policy",
        choices=("safe", "overlap"),
        default="safe",
        help="SRT output policy: merge simultaneous windows safely or keep overlapping cues",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated smoke artifacts without prompting",
    )


def _add_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--with-hf-fallback",
        action="store_true",
        help="Also download the retained HF fallback model bundle.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "HF model ID to download when `--with-hf-fallback` is enabled "
            f"(default: {DEFAULT_MODEL_ID})"
        ),
    )
    parser.add_argument(
        "--gguf-model-id",
        default=DEFAULT_GGUF_MODEL_ID,
        help=(
            "GGUF model ID to download for the llama.cpp path "
            f"(default: {DEFAULT_GGUF_MODEL_ID})"
        ),
    )
    parser.add_argument(
        "--with-qwen-corrector",
        action="store_true",
        help="Also download the retained local Qwen corrector assets.",
    )
    parser.add_argument(
        "--qwen-corrector-model-id",
        default=DEFAULT_QWEN_CORRECTOR_MODEL_ID,
        help=(
            "Qwen GGUF model ID to download for the optional local corrector path "
            f"(default: {DEFAULT_QWEN_CORRECTOR_MODEL_ID})"
        ),
    )
    parser.add_argument(
        "--qwen-corrector-model-filename",
        default=DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME,
        help=(
            "GGUF filename to download from the Qwen corrector repository "
            f"(default: {DEFAULT_QWEN_CORRECTOR_MODEL_FILENAME})"
        ),
    )
    parser.add_argument(
        "--qwen-corrector-mmproj-filename",
        default=DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME,
        help=(
            "mmproj filename to download from the Qwen corrector repository "
            f"(default: {DEFAULT_QWEN_CORRECTOR_MMPROJ_FILENAME})"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Local model cache root (default: ~/.cache/istots/models "
            "or ISTOTS_MODELS_DIR)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-materialize even when local cache already exists",
    )
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=None,
        help=(
            "Local support cache root for the optional pinned gguf-py snapshot fallback "
            "(default: ~/.cache/istots/support or ISTOTS_SUPPORT_DIR)."
        ),
    )
    parser.add_argument(
        "--gguf-py-base-url",
        default=None,
        help=(
            "Override source root for the optional pinned gguf-py snapshot fallback. "
            "Accepts an exact raw URL root or a local directory for offline setup."
        ),
    )
    parser.add_argument(
        "--gguf-source-mode",
        choices=("auto-download", "installed", "auto"),
        default="auto",
        help=(
            "How setup should source the gguf implementation while "
            "materializing the derived mmproj: auto (default: installed first, "
            "then pinned snapshot fallback), installed, or auto-download."
        ),
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=32768,
        help="clip.vision.image_min_pixels value for the derived llama.cpp mmproj (default: 32768)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )


def _add_auth_arguments(parser: argparse.ArgumentParser) -> None:
    auth_subparsers = parser.add_subparsers(dest="auth_provider", required=True)

    gemini = auth_subparsers.add_parser(
        "gemini",
        help="Manage Gemini API credentials",
    )
    gemini_subparsers = gemini.add_subparsers(dest="auth_action", required=True)
    gemini_subparsers.add_parser(
        "set",
        help="Store the Gemini API key in the local keyring",
    )
    gemini_subparsers.add_parser(
        "delete",
        help="Delete the Gemini API key from the local keyring",
    )
    gemini_subparsers.add_parser(
        "status",
        help="Show whether Gemini credentials are available",
    )

    env_file = gemini_subparsers.add_parser(
        "env-file",
        help="Manage the fallback Gemini .env file path",
    )
    env_file_subparsers = env_file.add_subparsers(dest="auth_env_file_action", required=True)
    env_file_set = env_file_subparsers.add_parser(
        "set",
        help="Configure the fallback Gemini .env file path",
    )
    env_file_set.add_argument("path", type=Path, help="Path to a Gemini .env file")
    env_file_subparsers.add_parser(
        "clear",
        help="Clear the configured Gemini .env file path",
    )


def _add_materialize_mmproj_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("base_mmproj", type=Path, help="Official base mmproj GGUF path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Derived mmproj output path (default: alongside base as *.minpix32768.gguf)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=32768,
        help="clip.vision.image_min_pixels value for the derived mmproj (default: 32768)",
    )
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=None,
        help=(
            "Local support cache root for the optional pinned gguf-py snapshot "
            "(default: ~/.cache/istots/support or ISTOTS_SUPPORT_DIR)."
        ),
    )
    parser.add_argument(
        "--gguf-py-base-url",
        default=None,
        help=(
            "Override source root for the optional pinned gguf-py snapshot fallback. "
            "Accepts an exact raw URL root or a local directory for offline setup."
        ),
    )
    parser.add_argument(
        "--gguf-source-mode",
        choices=("auto-download", "installed", "auto"),
        default="auto",
        help=(
            "How to source the gguf implementation: "
            "auto (default: installed first, then pinned auto-download fallback), "
            "installed, or auto-download."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the derived mmproj if it already exists",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )


def _add_doctor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "doctor_category",
        nargs="?",
        choices=("runtime", "auth", "workflow"),
        default=None,
        help=(
            "Structured doctor category. Use `runtime`, `auth`, or `workflow`."
        ),
    )
    parser.add_argument(
        "doctor_target",
        nargs="?",
        default=None,
        help=(
            "Structured doctor target. "
            "`runtime`: `paddle` or `qwen`; `auth`: `gemini`; "
            "`workflow`: `default`, `wider`, `corrector-qwen`, or `corrector-gemini`."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Local model cache root (default: ~/.cache/istots/models "
            "or ISTOTS_MODELS_DIR)."
        ),
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=32768,
        help="Derived mmproj min_pixels value used for fast-role asset resolution (default: 32768)",
    )
    parser.add_argument(
        "--llama-server-path",
        type=Path,
        default=None,
        help="Explicit llama-server binary path",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind or probe (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--input-sup",
        type=Path,
        default=None,
        help=(
            "Input .sup path for `doctor workflow ...`. Required for workflow doctor runs. "
            "Workflow temp artifacts are removed on success and retained on failure."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Gemini API key environment variable name for doctor auth/workflow checks.",
    )
    parser.add_argument(
        "--paddle-profile",
        choices=("auto", "cpu"),
        default="auto",
        help="PaddleOCR-VL runtime profile for `doctor runtime paddle` and `doctor workflow ...`.",
    )
    parser.add_argument(
        "--paddle-port",
        type=int,
        default=None,
        help="Override the shared PaddleOCR-VL llama-server port for structured doctor runs.",
    )
    parser.add_argument(
        "--paddle-threads",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server thread count for structured doctor runs.",
    )
    parser.add_argument(
        "--paddle-threads-batch",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server batch thread count for structured doctor runs.",
    )
    parser.add_argument(
        "--paddle-gpu-layers",
        type=int,
        default=None,
        help="Override PaddleOCR-VL llama-server GPU layer count for structured doctor runs.",
    )
    parser.add_argument(
        "--paddle-no-mmproj-offload",
        action="store_true",
        help="Disable mmproj offload for PaddleOCR-VL structured doctor runs.",
    )
    parser.add_argument(
        "--paddle-startup-timeout-sec",
        type=float,
        default=120.0,
        help="PaddleOCR-VL llama-server startup timeout for structured doctor runs.",
    )
    parser.add_argument(
        "--paddle-ctx-size",
        type=int,
        default=None,
        help=f"Override PaddleOCR-VL llama-server context size for structured doctor runs (default policy: {LOCAL_PADDLE_CTX_SIZE}).",
    )
    _add_temp_ocr_image_file_argument(parser, help_suffix="Applies to `doctor workflow ...` only.")
    parser.add_argument(
        "--corrector-model-path",
        type=Path,
        default=None,
        help="Explicit local GGUF corrector model path for `doctor runtime qwen` or `doctor workflow corrector-qwen`.",
    )
    parser.add_argument(
        "--corrector-mmproj-path",
        type=Path,
        default=None,
        help="Explicit local GGUF corrector mmproj path for `doctor runtime qwen` or `doctor workflow corrector-qwen`.",
    )
    parser.add_argument(
        "--qwen-profile",
        choices=("auto", "cpu"),
        default="auto",
        help="Qwen3.5 runtime profile for `doctor runtime qwen` and `doctor workflow corrector-qwen`.",
    )
    parser.add_argument(
        "--qwen-port",
        type=int,
        default=None,
        help="Override the Qwen3.5 llama-server port for structured doctor runs.",
    )
    parser.add_argument(
        "--qwen-threads",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server thread count for structured doctor runs.",
    )
    parser.add_argument(
        "--qwen-threads-batch",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server batch thread count for structured doctor runs.",
    )
    parser.add_argument(
        "--qwen-gpu-layers",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server GPU layer count for structured doctor runs.",
    )
    parser.add_argument(
        "--qwen-no-mmproj-offload",
        action="store_true",
        help="Force `--no-mmproj-offload` for structured Qwen doctor runs.",
    )
    parser.add_argument(
        "--qwen-ctx-size",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server context size for structured doctor runs.",
    )
    parser.add_argument(
        "--qwen-n-predict",
        type=int,
        default=None,
        help="Override Qwen3.5 llama-server `-n` value for structured doctor runs.",
    )
    parser.add_argument(
        "--qwen-reasoning",
        default=None,
        help="Override Qwen3.5 llama-server reasoning mode for structured doctor runs.",
    )
    parser.add_argument(
        "--qwen-startup-timeout-sec",
        type=float,
        default=120.0,
        help="Qwen3.5 llama-server startup timeout for structured doctor runs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv

    known_commands = {"convert", "setup", "materialize-mmproj", "doctor", "smoke", "auth"}
    first = argv[0]
    if first in known_commands or first.startswith("-"):
        return argv

    # Backward compatibility:
    # `istots input.sup output.srt` => `istots convert input.sup output.srt`
    return ["convert", *argv]


def run(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(raw_argv))

    if args.command == "setup":
        return run_setup(args)
    if args.command == "materialize-mmproj":
        return run_materialize_mmproj(args)
    if args.command == "doctor":
        return run_doctor(args)
    if args.command == "smoke":
        return run_smoke(args)
    if args.command == "auth":
        return run_auth(args)
    if args.command == "convert":
        return run_convert(args)

    parser.print_help()
    return 2


def run_setup(args: argparse.Namespace) -> int:
    configure_logging(verbose=not args.quiet)

    from istots.model_store import (
        is_default_pinned_gguf_model,
        is_default_pinned_hf_model,
        is_default_pinned_qwen_bundle,
        setup_default_runtime_assets,
    )

    if not args.with_hf_fallback and not is_default_pinned_hf_model(args.model_id):
        logging.getLogger(__name__).error(
            "setup failed: --model-id requires --with-hf-fallback"
        )
        return 1

    try:
        artifacts = setup_default_runtime_assets(
            hf_model_id=args.model_id,
            gguf_model_id=args.gguf_model_id,
            with_hf_fallback=args.with_hf_fallback,
            with_qwen_corrector=args.with_qwen_corrector,
            qwen_corrector_model_id=args.qwen_corrector_model_id,
            qwen_corrector_model_filename=args.qwen_corrector_model_filename,
            qwen_corrector_mmproj_filename=args.qwen_corrector_mmproj_filename,
            models_dir=args.models_dir,
            force=args.force,
            support_dir=args.support_dir,
            gguf_py_base_url=args.gguf_py_base_url,
            gguf_source_mode=args.gguf_source_mode,
            min_pixels=args.min_pixels,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("setup failed: %s", exc)
        return 1

    if not args.quiet:
        logger = logging.getLogger(__name__)
        if args.with_hf_fallback and not is_default_pinned_hf_model(args.model_id):
            logger.info(
                "HF fallback setup uses custom values; revision pinning and artifact hash "
                "verification remain user-managed for this bundle."
            )
        if not is_default_pinned_gguf_model(args.gguf_model_id):
            logger.info(
                "GGUF runtime setup uses custom values; revision pinning and artifact hash "
                "verification remain user-managed for this bundle."
            )
        if args.with_qwen_corrector and not is_default_pinned_qwen_bundle(
            model_id=args.qwen_corrector_model_id,
            model_filename=args.qwen_corrector_model_filename,
            mmproj_filename=args.qwen_corrector_mmproj_filename,
        ):
            logger.info(
                "Qwen corrector setup uses custom values; revision pinning and artifact hash "
                "verification remain user-managed for this bundle."
            )
        if artifacts.hf_model_dir is not None:
            logger.info("HF fallback model downloaded to: %s", artifacts.hf_model_dir)
        logger.info("GGUF runtime assets downloaded to: %s", artifacts.gguf_model_dir)
        logger.info("GGUF model path: %s", artifacts.gguf_model_path)
        logger.info("GGUF base mmproj path: %s", artifacts.gguf_mmproj_path)
        logger.info(
            "GGUF derived mmproj path: %s",
            artifacts.gguf_mmproj_minpix32768_path,
        )
        if artifacts.qwen_corrector_dir is not None:
            logger.info(
                "Qwen corrector assets downloaded to: %s",
                artifacts.qwen_corrector_dir,
            )
            logger.info(
                "Qwen corrector model path: %s",
                artifacts.qwen_corrector_model_path,
            )
            logger.info(
                "Qwen corrector mmproj path: %s",
                artifacts.qwen_corrector_mmproj_path,
            )
    return 0


def run_auth(args: argparse.Namespace) -> int:
    configure_logging(verbose=False)

    if args.auth_provider != "gemini":
        logging.getLogger(__name__).error("unsupported auth provider: %s", args.auth_provider)
        return 1

    from istots.gemini_auth import (
        clear_configured_gemini_env_file,
        delete_gemini_api_key,
        get_gemini_auth_status,
        set_configured_gemini_env_file,
        set_gemini_api_key,
    )

    try:
        if args.auth_action == "set":
            api_key = getpass.getpass("Gemini API key: ")
            backend_name = set_gemini_api_key(api_key)
            print(f"Gemini API key stored in keyring backend: {backend_name}")
            return 0
        if args.auth_action == "delete":
            backend_name = delete_gemini_api_key()
            if backend_name is not None:
                print(f"Gemini API key deleted from keyring backend: {backend_name}")
            else:
                print("Gemini API key deleted.")
            return 0
        if args.auth_action == "status":
            status = get_gemini_auth_status()
            print(
                "keyring: "
                + ("configured" if status.keyring_configured else "missing")
                + (
                    f" ({status.keyring_backend})"
                    if status.keyring_backend is not None
                    else " (unavailable)"
                )
            )
            if status.env_file_configured:
                print(f".env path: configured ({status.env_file_path})")
                print(".env key presence: " + ("configured" if status.env_file_contains_key else "missing"))
            else:
                print(".env path: missing")
            if status.process_env_configured:
                print(f"shell env: configured ({status.process_env_name})")
            else:
                print("shell env: missing")
            print(f"effective source: {status.effective_source or 'missing'}")
            return 0
        if args.auth_action == "env-file":
            if args.auth_env_file_action == "set":
                resolved = set_configured_gemini_env_file(args.path)
                print(f"Configured Gemini .env file: {resolved}")
                return 0
            if args.auth_env_file_action == "clear":
                clear_configured_gemini_env_file()
                print("Cleared the configured Gemini .env file path.")
                return 0
    except Exception as exc:
        logging.getLogger(__name__).error("auth command failed: %s", exc)
        return 1

    logging.getLogger(__name__).error("unsupported auth action")
    return 1


def run_materialize_mmproj(args: argparse.Namespace) -> int:
    configure_logging(verbose=not args.quiet)

    from istots.llama_mmproj import materialize_mmproj, read_mmproj_min_pixels

    try:
        output = materialize_mmproj(
            base_mmproj=args.base_mmproj,
            output_path=args.output,
            min_pixels=args.min_pixels,
            support_dir=args.support_dir,
            gguf_py_base_url=args.gguf_py_base_url,
            gguf_source_mode=args.gguf_source_mode,
            force=args.force,
        )
        applied_value = read_mmproj_min_pixels(
            output,
            support_dir=args.support_dir,
            gguf_py_base_url=args.gguf_py_base_url,
            gguf_source_mode=args.gguf_source_mode,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("mmproj materialization failed: %s", exc)
        return 1

    if not args.quiet:
        logging.getLogger(__name__).info(
            "materialized mmproj: %s (clip.vision.image_min_pixels=%s)",
            output,
            applied_value,
        )
    return 0


def _validate_doctor_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[str, str]:
    category = args.doctor_category
    target = args.doctor_target
    if category is None:
        parser.error("doctor requires a category: runtime, auth, or workflow")

    if target is None:
        parser.error("doctor category requires a target")

    normalized_target = str(target).strip().lower()
    allowed_targets = {
        "runtime": {"paddle", "qwen"},
        "auth": {"gemini"},
        "workflow": {"default", "wider", "corrector-qwen", "corrector-gemini"},
    }
    if normalized_target not in allowed_targets[category]:
        joined = ", ".join(sorted(allowed_targets[category]))
        parser.error(f"unsupported doctor target for {category}: {target!r}. Expected one of: {joined}")

    if category == "workflow" and args.input_sup is None:
        parser.error("doctor workflow requires --input-sup")

    return category, normalized_target


def _build_doctor_paddle_runtime_overrides(args: argparse.Namespace) -> PaddleOCRVLRuntimeOverrides:
    return PaddleOCRVLRuntimeOverrides(
        profile=args.paddle_profile,
        port=args.paddle_port,
        threads=args.paddle_threads,
        threads_batch=args.paddle_threads_batch,
        gpu_layers=args.paddle_gpu_layers,
        no_mmproj_offload=True if args.paddle_no_mmproj_offload else None,
        startup_timeout_sec=args.paddle_startup_timeout_sec,
        ctx_size=args.paddle_ctx_size,
    )


def _build_doctor_qwen_runtime_overrides(args: argparse.Namespace) -> Qwen35RuntimeOverrides:
    return Qwen35RuntimeOverrides(
        profile=args.qwen_profile,
        port=args.qwen_port,
        threads=args.qwen_threads,
        threads_batch=args.qwen_threads_batch,
        gpu_layers=args.qwen_gpu_layers,
        no_mmproj_offload=True if args.qwen_no_mmproj_offload else None,
        startup_timeout_sec=args.qwen_startup_timeout_sec,
        ctx_size=args.qwen_ctx_size,
        n_predict=args.qwen_n_predict,
        reasoning=args.qwen_reasoning,
    )


def _format_doctor_details(details: tuple[tuple[str, str], ...]) -> str:
    return " ".join(f"{key}={value}" for key, value in details)


def _log_doctor_suite_result(
    result,
    *,
    quiet: bool,
) -> int:
    logger = logging.getLogger(__name__)
    if result.ok:
        if not quiet:
            logger.info("doctor passed: category=%s target=%s", result.category, result.target)
            for check in result.checks:
                logger.info("doctor passed: check=%s %s", check.name, _format_doctor_details(check.details))
        return 0

    for check in result.checks:
        if check.ok:
            if not quiet:
                logger.info("doctor passed: check=%s %s", check.name, _format_doctor_details(check.details))
            continue
        for issue in check.issues:
            logger.error("doctor failed: check=%s [%s] %s", check.name, issue.code, issue.message)
        if not quiet and check.details:
            logger.error("doctor context: check=%s %s", check.name, _format_doctor_details(check.details))
    return 1


def run_doctor(args: argparse.Namespace) -> int:
    parser = _build_doctor_parser()
    category, target = _validate_doctor_mode(parser, args)
    configure_logging(verbose=not args.quiet)

    from istots.doctor import run_gemini_auth_doctor, run_paddle_runtime_doctor, run_qwen_runtime_doctor, run_workflow_doctor

    paddle_overrides = _build_doctor_paddle_runtime_overrides(args)
    qwen_overrides = _build_doctor_qwen_runtime_overrides(args)

    if category == "runtime" and target == "paddle":
        result = run_paddle_runtime_doctor(
            models_dir=args.models_dir,
            min_pixels=args.min_pixels,
            explicit_binary_path=args.llama_server_path,
            host=args.host,
            overrides=paddle_overrides,
            startup_timeout_sec=args.paddle_startup_timeout_sec,
        )
        return _log_doctor_suite_result(result, quiet=args.quiet)

    if category == "runtime" and target == "qwen":
        result = run_qwen_runtime_doctor(
            models_dir=args.models_dir,
            explicit_binary_path=args.llama_server_path,
            host=args.host,
            overrides=qwen_overrides,
            explicit_model_path=args.corrector_model_path,
            explicit_mmproj_path=args.corrector_mmproj_path,
            startup_timeout_sec=args.qwen_startup_timeout_sec,
        )
        return _log_doctor_suite_result(result, quiet=args.quiet)

    if category == "auth" and target == "gemini":
        result = run_gemini_auth_doctor(api_key_env=args.api_key_env)
        return _log_doctor_suite_result(result, quiet=args.quiet)

    if category == "workflow" and target is not None:
        result = run_workflow_doctor(
            workflow=target,
            input_sup=args.input_sup.expanduser().resolve(),
            models_dir=args.models_dir,
            min_pixels=args.min_pixels,
            explicit_binary_path=args.llama_server_path,
            host=args.host,
            paddle_overrides=paddle_overrides,
            qwen_overrides=qwen_overrides,
            explicit_qwen_model_path=args.corrector_model_path,
            explicit_qwen_mmproj_path=args.corrector_mmproj_path,
            api_key_env=args.api_key_env,
            startup_timeout_sec=max(args.paddle_startup_timeout_sec, args.qwen_startup_timeout_sec),
            use_temp_ocr_image_files=not args.no_temp_ocr_image_files,
        )
        return _log_doctor_suite_result(result, quiet=args.quiet)

    parser.error("unsupported doctor mode")
    return 2


def _validate_convert_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be a positive integer")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be a positive integer")
    if args.engine != "hf":
        if args.hf_device != "auto":
            parser.error("--hf-device is only valid with --engine hf")
        if args.hf_dtype != "auto":
            parser.error("--hf-dtype is only valid with --engine hf")
    if args.engine != "llama-server" and _has_paddle_runtime_override_request(args):
        parser.error("Paddle llama-server overrides are only valid with --engine llama-server")
    if args.detector_output is not None and args.engine != "llama-server":
        parser.error("--detector-output requires --engine llama-server")
    if args.detector_output is not None and args.ocr_mode != "default":
        parser.error("--detector-output requires --ocr-mode default")
    if args.detector_mode != "default" and args.engine != "llama-server":
        parser.error("--detector-mode requires --engine llama-server")
    if args.detector_mode != "default" and args.ocr_mode != "default":
        parser.error("--detector-mode requires --ocr-mode default")
    if args.detector_mode != "default" and args.detector_output is None and args.corrector == "off":
        parser.error("--detector-mode requires --detector-output or --corrector")
    if args.detector_family_addon and args.engine != "llama-server":
        parser.error("--detector-family-addon requires --engine llama-server")
    if args.detector_family_addon and args.ocr_mode != "default":
        parser.error("--detector-family-addon requires --ocr-mode default")
    if args.detector_family_addon and args.detector_output is None and args.corrector == "off":
        parser.error("--detector-family-addon requires --detector-output or --corrector")
    if args.corrector != "off" and args.engine != "llama-server":
        parser.error("--corrector requires --engine llama-server")
    if args.corrector != "off" and args.ocr_mode != "default":
        parser.error("--corrector requires --ocr-mode default")
    if args.corrector == "off" and args.corrector_output is not None:
        parser.error("--corrector-output requires --corrector")
    if args.corrector != "qwen-local" and _has_qwen_runtime_override_request(args):
        parser.error("Qwen llama-server overrides are only valid with --corrector qwen-local")
    if args.corrector == "qwen-local":
        has_model_path = args.corrector_model_path is not None
        has_mmproj_path = args.corrector_mmproj_path is not None
        if has_model_path != has_mmproj_path:
            parser.error(
                "--corrector qwen-local requires both --corrector-model-path and "
                "--corrector-mmproj-path when either is provided"
            )
    if args.corrector == "gemini":
        if args.corrector_model_path is not None or args.corrector_mmproj_path is not None:
            parser.error("--corrector-model-path and --corrector-mmproj-path are only valid with --corrector qwen-local")


def _has_paddle_runtime_override_request(args: argparse.Namespace) -> bool:
    return any(
        (
            args.paddle_profile != "auto",
            args.paddle_port is not None,
            args.paddle_threads is not None,
            args.paddle_threads_batch is not None,
            args.paddle_gpu_layers is not None,
            args.paddle_no_mmproj_offload,
            args.paddle_startup_timeout_sec != 120.0,
        )
    )


def _has_qwen_runtime_override_request(args: argparse.Namespace) -> bool:
    return any(
        (
            args.qwen_profile != "auto",
            args.qwen_port is not None,
            args.qwen_threads is not None,
            args.qwen_threads_batch is not None,
            args.qwen_gpu_layers is not None,
            args.qwen_no_mmproj_offload,
            args.qwen_ctx_size is not None,
            args.qwen_n_predict is not None,
            args.qwen_reasoning is not None,
            args.qwen_startup_timeout_sec != 120.0,
        )
    )


def _remove_temp_artifact_dir(output_dir: Path, *, label: str) -> None:
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"failed to remove {label} temporary artifacts at {output_dir}: {exc}") from exc


def _validate_smoke_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.detector_mode != "default" and args.no_detector and args.corrector == "off":
        parser.error(
            f"--detector-mode {args.detector_mode} requires detector-enabled smoke validation; "
            "remove --no-detector, keep --ocr-mode default, or enable --corrector"
        )
    if args.detector_family_addon and args.no_detector and args.corrector == "off":
        parser.error(
            "--detector-family-addon requires detector-enabled smoke validation; "
            "remove --no-detector, keep --ocr-mode default, or enable --corrector"
        )


def run_smoke(args: argparse.Namespace) -> int:
    parser = _build_smoke_parser()

    if args.input_sup is None:
        parser.error("--input-sup is required for smoke")
    _validate_smoke_args(parser, args)

    is_auto_output_dir = args.output_dir is None
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and not output_dir.is_dir():
            parser.error("--output-dir must be a directory path")
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="istots-smoke-")).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    input_sup = args.input_sup.expanduser().resolve()
    output_srt = output_dir / f"{input_sup.stem}.smoke.srt"
    detector_output = None
    if args.ocr_mode == "default" and not args.no_detector:
        detector_output = output_dir / f"{input_sup.stem}.detector.jsonl"
    corrector_output = None
    if args.corrector != "off":
        corrector_output = output_dir / f"{input_sup.stem}.corrected.jsonl"

    convert_args = argparse.Namespace(
        input_sup=input_sup,
        output_srt=output_srt,
        engine="llama-server",
        hf_device="auto",
        hf_dtype="auto",
        model_id=DEFAULT_MODEL_ID,
        models_dir=args.models_dir,
        max_items=None,
        max_new_tokens=args.max_new_tokens,
        ocr_mode=args.ocr_mode,
        paddle_profile=args.paddle_profile,
        llama_server_path=args.llama_server_path,
        paddle_port=args.paddle_port,
        paddle_threads=args.paddle_threads,
        paddle_threads_batch=args.paddle_threads_batch,
        paddle_gpu_layers=args.paddle_gpu_layers,
        paddle_no_mmproj_offload=args.paddle_no_mmproj_offload,
        paddle_startup_timeout_sec=args.paddle_startup_timeout_sec,
        paddle_ctx_size=args.paddle_ctx_size,
        quiet=args.quiet,
        furigana_mask=args.furigana_mask,
        no_temp_ocr_image_files=args.no_temp_ocr_image_files,
        detector_output=detector_output,
        detector_mode=args.detector_mode,
        detector_family_addon=args.detector_family_addon,
        corrector=args.corrector,
        corrector_output=corrector_output,
        corrector_model_path=args.corrector_model_path,
        corrector_mmproj_path=args.corrector_mmproj_path,
        qwen_profile=args.qwen_profile,
        qwen_port=args.qwen_port,
        qwen_threads=args.qwen_threads,
        qwen_threads_batch=args.qwen_threads_batch,
        qwen_gpu_layers=args.qwen_gpu_layers,
        qwen_no_mmproj_offload=args.qwen_no_mmproj_offload,
        qwen_ctx_size=args.qwen_ctx_size,
        qwen_n_predict=args.qwen_n_predict,
        qwen_reasoning=args.qwen_reasoning,
        qwen_startup_timeout_sec=args.qwen_startup_timeout_sec,
        corrector_gemini_model=args.corrector_gemini_model,
        corrector_api_key_env=args.corrector_api_key_env,
        corrector_thinking_level=args.corrector_thinking_level,
        corrector_media_resolution=args.corrector_media_resolution,
        corrector_cache_dir=args.corrector_cache_dir,
        corrector_gemini_max_attempts=args.corrector_gemini_max_attempts,
        corrector_gemini_request_timeout_sec=args.corrector_gemini_request_timeout_sec,
        corrector_gemini_max_workers=args.corrector_gemini_max_workers,
        srt_policy=args.srt_policy,
        force=args.force,
    )
    rc = _run_convert_impl(convert_args, parser)
    if rc != 0:
        if is_auto_output_dir:
            logging.getLogger(__name__).error("retained temporary smoke artifacts at %s", output_dir)
        return rc
    if not is_auto_output_dir:
        return rc
    try:
        _remove_temp_artifact_dir(output_dir, label="smoke")
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    if not args.quiet:
        logging.getLogger(__name__).info("removed temporary smoke artifacts after success: %s", output_dir)
    return 0


def run_convert(args: argparse.Namespace) -> int:
    parser = _build_convert_parser()
    return _run_convert_impl(args, parser)


def _validate_distinct_convert_paths(
    parser: argparse.ArgumentParser,
    *,
    input_sup: Path,
    output_srt: Path,
    detector_output: Path | None,
    corrector_output: Path | None,
) -> None:
    seen_paths: dict[Path, str] = {}
    for label, path in (
        ("input_sup", input_sup),
        ("output_srt", output_srt),
        ("detector_output", detector_output),
        ("corrector_output", corrector_output),
    ):
        if path is None:
            continue
        previous_label = seen_paths.get(path)
        if previous_label is not None:
            parser.error(f"{previous_label} and {label} must be different paths")
        seen_paths[path] = label


def _convert_output_artifacts(
    *,
    output_srt: Path,
    detector_output: Path | None,
    corrector_output: Path | None,
) -> tuple[Path, ...]:
    artifacts = [output_srt]
    if detector_output is not None:
        artifacts.append(detector_output)
    if corrector_output is not None:
        artifacts.append(corrector_output)
    return tuple(artifacts)


def _check_existing_convert_outputs(
    *,
    output_srt: Path,
    detector_output: Path | None,
    corrector_output: Path | None,
    force: bool,
) -> int:
    existing_paths = [path for path in _convert_output_artifacts(
        output_srt=output_srt,
        detector_output=detector_output,
        corrector_output=corrector_output,
    ) if path.exists()]
    if not existing_paths or force:
        return 0

    logger = logging.getLogger(__name__)
    if _can_prompt_for_overwrite():
        for path in existing_paths:
            if not _confirm_overwrite(path):
                logger.error("conversion cancelled")
                return 1
        return 0

    if len(existing_paths) == 1:
        logger.error(
            "output artifact already exists: %s. Rerun with --force to overwrite.",
            existing_paths[0],
        )
    else:
        logger.error(
            "output artifacts already exist: %s. Rerun with --force to overwrite.",
            ", ".join(str(path) for path in existing_paths),
        )
    return 1


def _run_convert_impl(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _validate_convert_args(parser, args)

    configure_logging(verbose=not args.quiet)

    input_sup = args.input_sup.expanduser().resolve()
    output_srt = args.output_srt.expanduser().resolve()
    detector_output = args.detector_output.expanduser().resolve() if args.detector_output is not None else None
    corrector_output = (
        args.corrector_output.expanduser().resolve() if args.corrector_output is not None else None
    )

    if output_srt.exists() and output_srt.is_dir():
        parser.error("output_srt must be a file path, not an existing directory")
    if detector_output is not None and detector_output.exists() and detector_output.is_dir():
        parser.error("detector_output must be a file path, not an existing directory")
    if corrector_output is not None and corrector_output.exists() and corrector_output.is_dir():
        parser.error("corrector_output must be a file path, not an existing directory")
    _validate_distinct_convert_paths(
        parser,
        input_sup=input_sup,
        output_srt=output_srt,
        detector_output=detector_output,
        corrector_output=corrector_output,
    )
    overwrite_check = _check_existing_convert_outputs(
        output_srt=output_srt,
        detector_output=detector_output,
        corrector_output=corrector_output,
        force=args.force,
    )
    if overwrite_check != 0:
        return overwrite_check
    if args.corrector_gemini_max_attempts < 1:
        parser.error("--corrector-gemini-max-attempts must be >= 1")
    if args.corrector_gemini_request_timeout_sec <= 0:
        parser.error("--corrector-gemini-request-timeout-sec must be > 0")
    if args.corrector_gemini_max_workers < 1:
        parser.error("--corrector-gemini-max-workers must be >= 1")

    from istots.model_store import ensure_local_model, ensure_local_qwen_corrector_assets
    from istots.pipeline import convert_sup_to_srt
    from istots.ocr import PaddleOCRVLRuntimeOverrides, Qwen35RuntimeOverrides

    corrector_config = None
    if args.corrector != "off":
        from istots.corrector import CorrectorConfig, CorrectorMode

        resolved_corrector_model_path = None
        resolved_corrector_mmproj_path = None
        if args.corrector == "qwen-local":
            if args.corrector_model_path is not None and args.corrector_mmproj_path is not None:
                resolved_corrector_model_path = args.corrector_model_path.expanduser().resolve()
                resolved_corrector_mmproj_path = args.corrector_mmproj_path.expanduser().resolve()
            else:
                try:
                    (
                        resolved_corrector_model_path,
                        resolved_corrector_mmproj_path,
                    ) = ensure_local_qwen_corrector_assets(models_dir=args.models_dir)
                except Exception as exc:
                    logging.getLogger(__name__).error("corrector asset check failed: %s", exc)
                    return 1
        corrector_config = CorrectorConfig(
            mode=CorrectorMode(args.corrector),
            output_path=corrector_output,
            local_model_path=resolved_corrector_model_path,
            local_mmproj_path=resolved_corrector_mmproj_path,
            local_runtime_overrides=Qwen35RuntimeOverrides(
                profile=args.qwen_profile,
                port=args.qwen_port,
                threads=args.qwen_threads,
                threads_batch=args.qwen_threads_batch,
                gpu_layers=args.qwen_gpu_layers,
                no_mmproj_offload=True if args.qwen_no_mmproj_offload else None,
                startup_timeout_sec=args.qwen_startup_timeout_sec,
                ctx_size=args.qwen_ctx_size,
                n_predict=args.qwen_n_predict,
                reasoning=args.qwen_reasoning,
            ),
            api_key_env=args.corrector_api_key_env,
            gemini_model=args.corrector_gemini_model,
            thinking_level=args.corrector_thinking_level,
            media_resolution=args.corrector_media_resolution,
            cache_dir=(
                args.corrector_cache_dir.expanduser().resolve()
                if args.corrector_cache_dir is not None
                else None
            ),
            max_attempts=args.corrector_gemini_max_attempts,
            request_timeout=args.corrector_gemini_request_timeout_sec,
            gemini_max_workers=args.corrector_gemini_max_workers,
        )

    model_id = args.model_id
    if args.engine == "hf":
        try:
            model_path = ensure_local_model(
                model_id=args.model_id,
                models_dir=args.models_dir,
            )
        except Exception as exc:
            logging.getLogger(__name__).error("model check failed: %s", exc)
            return 1
        model_id = str(model_path)
        if not args.quiet:
            logging.getLogger(__name__).info("using HF fallback model: %s", model_path)
    elif not args.quiet:
        logging.getLogger(__name__).info(
            "using primary OCR engine: %s (mode=%s profile=%s)",
            args.engine,
            args.ocr_mode,
            args.paddle_profile,
        )
        if corrector_config is not None:
            logging.getLogger(__name__).info("using conservative corrector: %s", corrector_config.mode)
            if corrector_config.mode is CorrectorMode.QWEN_LOCAL:
                logging.getLogger(__name__).info(
                    "using local Qwen corrector assets: model=%s mmproj=%s",
                    corrector_config.local_model_path,
                    corrector_config.local_mmproj_path,
                )

    paddle_runtime_overrides = PaddleOCRVLRuntimeOverrides(
        profile=args.paddle_profile,
        port=args.paddle_port,
        threads=args.paddle_threads,
        threads_batch=args.paddle_threads_batch,
        gpu_layers=args.paddle_gpu_layers,
        no_mmproj_offload=True if args.paddle_no_mmproj_offload else None,
        startup_timeout_sec=args.paddle_startup_timeout_sec,
        ctx_size=args.paddle_ctx_size,
    )

    try:
        result = convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            hf_device=args.hf_device,
            hf_dtype=args.hf_dtype,
            engine=args.engine,
            ocr_mode=args.ocr_mode,
            detector_output=detector_output,
            detector_mode=args.detector_mode,
            detector_family_addon=args.detector_family_addon,
            corrector_config=corrector_config,
            model_id=model_id,
            models_dir=args.models_dir,
            max_items=args.max_items,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.engine == "hf",
            enable_furigana_mask=args.furigana_mask,
            srt_policy=args.srt_policy,
            runtime_binary_path=args.llama_server_path,
            paddle_runtime_overrides=paddle_runtime_overrides,
            use_temp_ocr_image_files=not args.no_temp_ocr_image_files,
            verbose=not args.quiet,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("conversion failed: %s", exc)
        return 1

    if not args.quiet:
        if args.engine == "hf":
            logging.getLogger(__name__).info(
                "done: wrote %d subtitles to %s (hf-device=%s)",
                result.written_count,
                result.output_srt,
                result.device_used,
            )
        else:
            logging.getLogger(__name__).info(
                "done: wrote %d subtitles to %s (llama-profile=%s)",
                result.written_count,
                result.output_srt,
                result.device_used,
            )
        if detector_output is not None:
            logging.getLogger(__name__).info(
                "detector manifest: %s disagreements=%d",
                detector_output,
                result.detector_record_count,
            )
        if corrector_config is not None:
            fallback_count = getattr(result, "correction_fallback_count", 0)
            logging.getLogger(__name__).info(
                "conservative correction: rows=%d applied=%d fallback=%d",
                result.correction_record_count,
                result.correction_applied_count,
                fallback_count,
            )
            if corrector_output is not None:
                logging.getLogger(__name__).info("corrector manifest: %s", corrector_output)
    return 0


def _can_prompt_for_overwrite() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _confirm_overwrite(output_path: Path) -> bool:
    response = input(f"output artifact already exists: {output_path}\noverwrite? [y/N]: ")
    return response.strip().lower() in {"y", "yes"}


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main() -> None:
    raise SystemExit(run())
