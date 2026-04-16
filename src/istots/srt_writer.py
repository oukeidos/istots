from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from istots.atomic_writer import atomic_write_text_file


@dataclass
class SubtitleEntry:
    index: int
    start: timedelta
    end: timedelta
    text: str


def write_srt(entries: Iterable[SubtitleEntry], output_srt: Path) -> None:
    def write_entries(fp) -> None:
        for entry in entries:
            fp.write(f"{entry.index}\n")
            fp.write(f"{format_timestamp(entry.start)} --> {format_timestamp(entry.end)}\n")
            fp.write(f"{entry.text}\n\n")

    atomic_write_text_file(output_srt, write_entries, encoding="utf-8", newline="\n")


def format_timestamp(ts: timedelta) -> str:
    total_ms = int(max(ts.total_seconds(), 0.0) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
