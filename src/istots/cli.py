from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from istots import __version__
from istots.model_store import DEFAULT_MODEL_ID

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

    return parser


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_sup", type=Path, help="Input .sup file")
    parser.add_argument("output_srt", type=Path, help="Output .srt file")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device selection (default: auto)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "Model ID or local model path. If model ID is given, it must already "
            "exist in local cache from `istots setup`."
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
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
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
        help=f"Model ID to download (default: {DEFAULT_MODEL_ID})",
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
        help="Re-download even when local cache already exists",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv

    known_commands = {"convert", "setup"}
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
    if args.command == "convert":
        return run_convert(args)

    parser.print_help()
    return 2


def run_setup(args: argparse.Namespace) -> int:
    configure_logging(verbose=not args.quiet)

    from istots.model_store import download_model

    try:
        path = download_model(
            model_id=args.model_id,
            models_dir=args.models_dir,
            force=args.force,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("setup failed: %s", exc)
        return 1

    if not args.quiet:
        logging.getLogger(__name__).info("model downloaded to: %s", path)
    return 0


def run_convert(args: argparse.Namespace) -> int:
    parser = build_parser()

    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be a positive integer")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    configure_logging(verbose=not args.quiet)

    input_sup = args.input_sup.expanduser().resolve()
    output_srt = args.output_srt.expanduser().resolve()

    if output_srt.exists() and output_srt.is_dir():
        parser.error("output_srt must be a file path, not an existing directory")
    if input_sup == output_srt:
        parser.error("input_sup and output_srt must be different paths")
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

    try:
        model_path = ensure_local_model(
            model_id=args.model_id,
            models_dir=args.models_dir,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("model check failed: %s", exc)
        return 1

    if not args.quiet:
        logging.getLogger(__name__).info("using local model: %s", model_path)

    try:
        result = convert_sup_to_srt(
            input_sup=input_sup,
            output_srt=output_srt,
            preferred_device=args.device,
            model_id=str(model_path),
            max_items=args.max_items,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            local_files_only=True,
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
