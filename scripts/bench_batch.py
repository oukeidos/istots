from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from istots.device import resolve_device
from istots.model_store import DEFAULT_MODEL_ID, ensure_local_model
from istots.ocr.hf_backend import HFPaddleOCRVLBackend
from istots.sup_reader import SubtitleFrame, iter_sup_frames


@dataclass
class BatchRun:
    batch_size: int
    repeat: int
    ok: bool
    items: int
    elapsed_sec: float
    items_per_sec: float
    peak_vram_mib: float | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR throughput by batch size on a subset of SUP frames.",
    )
    parser.add_argument("input_sup", type=Path, help="Input .sup file")
    parser.add_argument(
        "--max-items",
        type=int,
        default=240,
        help="Use only first N subtitle frames (default: 240)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="4,8,12,16,24,32,40,48,56,64",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of measured repeats per batch size (default: 1)",
    )
    parser.add_argument(
        "--warmup-items",
        type=int,
        default=24,
        help="Warm up with first N frames before measurement (default: 24)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device selection (default: auto)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Model ID or local path (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Local model cache root (default: ~/.cache/istots/models)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per frame (default: 256)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path for raw run data",
    )
    return parser.parse_args()


def parse_batch_sizes(spec: str) -> list[int]:
    seen: set[int] = set()
    values: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("batch size must be positive")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("no valid batch size specified")
    return values


def chunked(items: Sequence[SubtitleFrame], size: int) -> Iterable[Sequence[SubtitleFrame]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def reset_peak_vram(backend: HFPaddleOCRVLBackend) -> None:
    torch_mod = getattr(backend, "_torch", None)
    if backend.device != "cuda" or torch_mod is None or not hasattr(torch_mod, "cuda"):
        return
    if hasattr(torch_mod.cuda, "reset_peak_memory_stats"):
        torch_mod.cuda.reset_peak_memory_stats()


def read_peak_vram_mib(backend: HFPaddleOCRVLBackend) -> float | None:
    torch_mod = getattr(backend, "_torch", None)
    if backend.device != "cuda" or torch_mod is None or not hasattr(torch_mod, "cuda"):
        return None
    if not hasattr(torch_mod.cuda, "max_memory_allocated"):
        return None
    return float(torch_mod.cuda.max_memory_allocated()) / (1024.0 * 1024.0)


def run_once(
    backend: HFPaddleOCRVLBackend,
    frames: Sequence[SubtitleFrame],
    batch_size: int,
    repeat: int,
) -> BatchRun:
    reset_peak_vram(backend)
    started = time.perf_counter()
    processed = 0

    try:
        for group in chunked(frames, batch_size):
            texts = backend.recognize_batch([frame.image for frame in group])
            if len(texts) != len(group):
                raise RuntimeError(
                    f"mismatched OCR result size: expected {len(group)}, got {len(texts)}"
                )
            processed += len(group)
        elapsed = time.perf_counter() - started
        throughput = processed / elapsed if elapsed > 0 else 0.0
        return BatchRun(
            batch_size=batch_size,
            repeat=repeat,
            ok=True,
            items=processed,
            elapsed_sec=elapsed,
            items_per_sec=throughput,
            peak_vram_mib=read_peak_vram_mib(backend),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return BatchRun(
            batch_size=batch_size,
            repeat=repeat,
            ok=False,
            items=processed,
            elapsed_sec=elapsed,
            items_per_sec=0.0,
            peak_vram_mib=read_peak_vram_mib(backend),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        backend.clear_device_cache()
        gc.collect()


def warmup(backend: HFPaddleOCRVLBackend, frames: Sequence[SubtitleFrame], warmup_items: int, batch_size: int) -> None:
    if warmup_items <= 0 or not frames:
        return
    subset = frames[: min(warmup_items, len(frames))]
    for group in chunked(subset, max(1, batch_size)):
        backend.recognize_batch([frame.image for frame in group])
    backend.clear_device_cache()
    gc.collect()


def print_summary(results: Sequence[BatchRun]) -> None:
    grouped: dict[int, list[BatchRun]] = {}
    for row in results:
        grouped.setdefault(row.batch_size, []).append(row)

    print("\n=== Summary (by batch size) ===")
    print("batch\tok/total\tmean_items/s\tbest_items/s\tmean_time(s)\tmean_peak_vram(MiB)")

    ranking: list[tuple[float, int]] = []
    for batch_size in sorted(grouped):
        runs = grouped[batch_size]
        ok_runs = [r for r in runs if r.ok]
        if not ok_runs:
            print(f"{batch_size}\t0/{len(runs)}\t-\t-\t-\t-")
            continue

        mean_ips = statistics.fmean(r.items_per_sec for r in ok_runs)
        best_ips = max(r.items_per_sec for r in ok_runs)
        mean_time = statistics.fmean(r.elapsed_sec for r in ok_runs)
        vram_values = [r.peak_vram_mib for r in ok_runs if r.peak_vram_mib is not None]
        mean_vram = statistics.fmean(vram_values) if vram_values else None

        ranking.append((mean_ips, batch_size))
        vram_text = f"{mean_vram:.1f}" if mean_vram is not None else "-"
        print(
            f"{batch_size}\t{len(ok_runs)}/{len(runs)}\t"
            f"{mean_ips:.2f}\t{best_ips:.2f}\t{mean_time:.2f}\t{vram_text}"
        )

    if not ranking:
        print("\nNo successful runs.")
        return

    ranking.sort(reverse=True)
    best_ips, best_batch = ranking[0]
    print("\n=== Fastest ===")
    print(f"batch={best_batch}, mean_items/s={best_ips:.2f}")

    print("\n=== Top 5 ===")
    for idx, (ips, batch_size) in enumerate(ranking[:5], start=1):
        print(f"{idx}. batch={batch_size}, mean_items/s={ips:.2f}")


def main() -> int:
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    if args.max_items <= 0:
        raise ValueError("--max-items must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    input_sup = args.input_sup.expanduser().resolve()
    if not input_sup.exists():
        raise FileNotFoundError(f"input not found: {input_sup}")

    model_path = ensure_local_model(args.model_id, models_dir=args.models_dir)
    device = resolve_device(args.device)

    print(f"input: {input_sup}")
    print(f"device: {device}")
    print(f"model: {model_path}")
    print(f"max_items: {args.max_items}")
    print(f"batch_sizes: {batch_sizes}")
    print("loading SUP frames...")

    frames = list(iter_sup_frames(input_sup, max_items=args.max_items))
    if not frames:
        raise RuntimeError("no frames parsed from SUP")
    print(f"parsed_frames: {len(frames)}")

    backend = HFPaddleOCRVLBackend(
        model_id=str(model_path),
        device=device,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
    )

    runs: list[BatchRun] = []
    try:
        print("warming up...")
        warmup(backend, frames, args.warmup_items, batch_sizes[0])
        print("benchmarking...")

        for batch_size in batch_sizes:
            for repeat in range(1, args.repeats + 1):
                print(f"run: batch={batch_size}, repeat={repeat}/{args.repeats}")
                row = run_once(backend, frames, batch_size=batch_size, repeat=repeat)
                runs.append(row)
                if row.ok:
                    vram_text = f"{row.peak_vram_mib:.1f} MiB" if row.peak_vram_mib is not None else "-"
                    print(
                        f"  ok: elapsed={row.elapsed_sec:.2f}s, items/s={row.items_per_sec:.2f}, "
                        f"peak_vram={vram_text}"
                    )
                else:
                    print(f"  fail: elapsed={row.elapsed_sec:.2f}s, error={row.error}")
    finally:
        backend.close()

    print_summary(runs)

    if args.json_out is not None:
        payload = [asdict(row) for row in runs]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"\nraw_json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
