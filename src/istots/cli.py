from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from istots import __version__
from istots.model_store import DEFAULT_GGUF_MODEL_ID, DEFAULT_MODEL_ID

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
        help="Run runtime preflight checks for retained runtime roles",
    )
    parser.register_subcommand_parser(doctor)
    _add_doctor_arguments(doctor)

    return parser


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_sup", type=Path, help="Input .sup file")
    parser.add_argument("output_srt", type=Path, help="Output .srt file")
    parser.add_argument(
        "--engine",
        choices=("llama-server", "hf"),
        default="llama-server",
        help="OCR engine selection (default: llama-server)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Device selection (default: auto)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "HF model ID or local HF model path for `--engine hf`. "
            "If a model ID is given, it must already exist in local cache from `istots setup`."
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
        "--batch-size",
        type=int,
        default=1,
        help="OCR batch size (default: 1)",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=("default", "fast"),
        default="default",
        help=(
            "OCR mode for `--engine llama-server`: retained default OCR or the optional "
            "fast hybrid path (default: default)"
        ),
    )
    parser.add_argument(
        "--runtime-profile",
        choices=("auto", "cpu", "memory"),
        default="auto",
        help="llama-server runtime profile when using `--engine llama-server` (default: auto)",
    )
    parser.add_argument(
        "--llama-server-path",
        type=Path,
        default=None,
        help="Explicit llama-server binary path for `--engine llama-server`",
    )
    parser.add_argument(
        "--runtime-port",
        type=int,
        default=None,
        help="Override the retained llama-server port for `--engine llama-server`",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override llama-server thread count for `--engine llama-server`",
    )
    parser.add_argument(
        "--threads-batch",
        type=int,
        default=None,
        help="Override llama-server batch thread count for `--engine llama-server`",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="Override llama-server GPU layer count for `--engine llama-server`",
    )
    parser.add_argument(
        "--no-mmproj-offload",
        action="store_true",
        help="Disable mmproj offload for `--engine llama-server`",
    )
    parser.add_argument(
        "--startup-timeout-sec",
        type=float,
        default=120.0,
        help="llama-server startup timeout in seconds for `--engine llama-server`",
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
        "--srt-policy",
        choices=("safe", "overlap"),
        default="safe",
        help="SRT output policy: merge simultaneous windows safely or keep overlapping cues",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file without prompting",
    )


def _add_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HF model ID to download for the retained fallback path (default: {DEFAULT_MODEL_ID})",
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
            "Local support cache root for pinned gguf-py snapshot fallback "
            "(default: ~/.cache/istots/support or ISTOTS_SUPPORT_DIR)."
        ),
    )
    parser.add_argument(
        "--gguf-py-base-url",
        default=None,
        help=(
            "Override source root for the pinned gguf-py snapshot fallback. "
            "Accepts an exact raw URL root or a local directory for offline setup."
        ),
    )
    parser.add_argument(
        "--gguf-source-mode",
        choices=("auto-download", "installed", "auto"),
        default="auto",
        help=(
            "How setup should source the known-good gguf implementation while "
            "materializing the derived mmproj: auto (default), installed, or auto-download."
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
            "Local support cache root for pinned gguf-py snapshot "
            "(default: ~/.cache/istots/support or ISTOTS_SUPPORT_DIR)."
        ),
    )
    parser.add_argument(
        "--gguf-py-base-url",
        default=None,
        help=(
            "Override source root for the pinned gguf-py snapshot. "
            "Accepts an exact raw URL root or a local directory for offline setup."
        ),
    )
    parser.add_argument(
        "--gguf-source-mode",
        choices=("auto-download", "installed", "auto"),
        default="auto",
        help=(
            "How to source the known-good gguf implementation: "
            "auto (default: installed first, then exact pinned auto-download fallback), "
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
        "--engine",
        choices=("llama-server",),
        default="llama-server",
        help="Runtime engine to validate (default: llama-server)",
    )
    parser.add_argument(
        "--role",
        choices=("ocr", "ocr-fast", "detector", "corrector"),
        default="ocr",
        help="Retained runtime role to validate (default: ocr)",
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "cpu", "memory"),
        default="auto",
        help="Runtime launch profile (default: auto)",
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
        "--port",
        type=int,
        default=None,
        help="Override the retained default port for the selected role",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=None,
        help="Override the runtime device preference",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override llama-server thread count",
    )
    parser.add_argument(
        "--threads-batch",
        type=int,
        default=None,
        help="Override llama-server batch thread count",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="Override llama-server GPU layer count",
    )
    parser.add_argument(
        "--no-mmproj-offload",
        action="store_true",
        help="Disable mmproj offload for the doctor launch",
    )
    parser.add_argument(
        "--startup-timeout-sec",
        type=float,
        default=120.0,
        help="llama-server startup timeout in seconds",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv

    known_commands = {"convert", "setup", "materialize-mmproj", "doctor"}
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
    if args.command == "convert":
        return run_convert(args)

    parser.print_help()
    return 2


def run_setup(args: argparse.Namespace) -> int:
    configure_logging(verbose=not args.quiet)

    from istots.model_store import setup_default_runtime_assets

    try:
        artifacts = setup_default_runtime_assets(
            hf_model_id=args.model_id,
            gguf_model_id=args.gguf_model_id,
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
        logging.getLogger(__name__).info("HF fallback model downloaded to: %s", artifacts.hf_model_dir)
        logging.getLogger(__name__).info("GGUF runtime assets downloaded to: %s", artifacts.gguf_model_dir)
        logging.getLogger(__name__).info("GGUF model path: %s", artifacts.gguf_model_path)
        logging.getLogger(__name__).info("GGUF base mmproj path: %s", artifacts.gguf_mmproj_path)
        logging.getLogger(__name__).info(
            "GGUF derived mmproj path: %s",
            artifacts.gguf_mmproj_minpix32768_path,
        )
    return 0


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


def run_doctor(args: argparse.Namespace) -> int:
    configure_logging(verbose=not args.quiet)

    if args.engine != "llama-server":
        logging.getLogger(__name__).error("unsupported doctor engine: %s", args.engine)
        return 1

    from istots.llama_runtime import LlamaServerOverrides, LlamaServerProfile, run_llama_server_doctor

    overrides = LlamaServerOverrides(
        profile=LlamaServerProfile(args.profile),
        device=args.device,
        threads=args.threads,
        threads_batch=args.threads_batch,
        port=args.port,
        gpu_layers=args.gpu_layers,
        no_mmproj_offload=True if args.no_mmproj_offload else None,
    )

    report = run_llama_server_doctor(
        role=args.role,
        models_dir=args.models_dir,
        min_pixels=args.min_pixels,
        explicit_binary_path=args.llama_server_path,
        host=args.host,
        overrides=overrides,
        startup_timeout_sec=args.startup_timeout_sec,
    )

    logger = logging.getLogger(__name__)
    if report.ok:
        if not args.quiet and report.launch_spec is not None:
            logger.info(
                "doctor passed: role=%s profile=%s binary=%s model=%s mmproj=%s port=%s",
                report.role,
                report.profile,
                report.launch_spec.binary_path,
                report.launch_spec.model_path,
                report.launch_spec.mmproj_path,
                report.launch_spec.port,
            )
            if report.smoke_response is not None:
                logger.info("doctor smoke response: %s", report.smoke_response)
        return 0

    for issue in report.issues:
        logger.error("doctor failed [%s]: %s", issue.code, issue.message)
    return 1


def run_convert(args: argparse.Namespace) -> int:
    parser = build_parser()

    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be a positive integer")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.ocr_mode == "fast" and args.engine != "llama-server":
        parser.error("--ocr-mode fast requires --engine llama-server")
    if args.ocr_mode == "fast" and args.runtime_port is not None:
        parser.error("--runtime-port is only supported with --ocr-mode default")
    if args.detector_output is not None and args.engine != "llama-server":
        parser.error("--detector-output requires --engine llama-server")
    if args.detector_output is not None and args.ocr_mode != "default":
        parser.error("--detector-output requires --ocr-mode default")

    configure_logging(verbose=not args.quiet)

    input_sup = args.input_sup.expanduser().resolve()
    output_srt = args.output_srt.expanduser().resolve()
    detector_output = args.detector_output.expanduser().resolve() if args.detector_output is not None else None

    if output_srt.exists() and output_srt.is_dir():
        parser.error("output_srt must be a file path, not an existing directory")
    if input_sup == output_srt:
        parser.error("input_sup and output_srt must be different paths")
    if detector_output is not None and detector_output.exists() and detector_output.is_dir():
        parser.error("detector_output must be a file path, not an existing directory")
    if output_srt.exists() and not args.force:
        if _can_prompt_for_overwrite():
            if not _confirm_overwrite(output_srt):
                logging.getLogger(__name__).error("conversion cancelled")
                return 1
        else:
            logging.getLogger(__name__).error(
                "output file already exists: %s. Rerun with --force to overwrite.",
                output_srt,
            )
            return 1

    from istots.model_store import ensure_local_model
    from istots.pipeline import convert_sup_to_srt

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
            args.runtime_profile,
        )

    try:
        result = convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            preferred_device=args.device,
            engine=args.engine,
            ocr_mode=args.ocr_mode,
            detector_output=detector_output,
            model_id=model_id,
            models_dir=args.models_dir,
            max_items=args.max_items,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.engine == "hf",
            enable_furigana_mask=args.furigana_mask,
            srt_policy=args.srt_policy,
            runtime_profile=args.runtime_profile,
            runtime_binary_path=args.llama_server_path,
            runtime_port=args.runtime_port,
            runtime_threads=args.threads,
            runtime_threads_batch=args.threads_batch,
            runtime_gpu_layers=args.gpu_layers,
            runtime_no_mmproj_offload=True if args.no_mmproj_offload else None,
            runtime_startup_timeout_sec=args.startup_timeout_sec,
            verbose=not args.quiet,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("conversion failed: %s", exc)
        return 1

    if not args.quiet:
        logging.getLogger(__name__).info(
            "done: wrote %d subtitles to %s (device=%s)",
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
    return 0


def _can_prompt_for_overwrite() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _confirm_overwrite(output_srt: Path) -> bool:
    response = input(f"output file already exists: {output_srt}\noverwrite? [y/N]: ")
    return response.strip().lower() in {"y", "yes"}


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main() -> None:
    raise SystemExit(run())
