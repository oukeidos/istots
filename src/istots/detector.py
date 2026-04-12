from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HybridDetectorRecord:
    index: int
    raw_index: int
    window_id: int
    start_ms: int
    end_ms: int
    detector_branch: str
    shape: str
    ratio: float
    option_role: str
    baseline_text: str
    option_text: str
    diff_label: str
    meaningful: bool
    char_error_rate: float
    source_tags: tuple[str, ...] = ()
    alternate_source_kind: str | None = None
    dominant_family: str | None = None
    family_current_char: str | None = None
    family_alternate_char: str | None = None
    family_support_rows: int | None = None
    family_pure_rows: int | None = None
    family_mixed_rows: int | None = None
    family_agreement_rows: int | None = None


def write_hybrid_detector_records(path: Path, records: list[HybridDetectorRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
