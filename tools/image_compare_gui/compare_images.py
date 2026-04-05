from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ZOOM_STEP = 1.25
MIN_ZOOM_MULTIPLIER = 0.2
MAX_ZOOM_MULTIPLIER = 8.0
BACKGROUND_COLOR = "#202124"
MISSING_COLOR = "#f28b82"
BASE_FONT_SIZE = 16
TITLE_FONT_SIZE = 18
MESSAGE_FONT_SIZE = 20
HELP_FONT_SIZE = 15
BUTTON_PADDING = (12, 8)


@dataclass(frozen=True)
class PanelSpec:
    label: str
    root: Path
    files: dict[str, Path]


@dataclass(frozen=True)
class EntryMetadata:
    relative_path: str
    frame_index: int | None
    segment_index: int | None
    window_id: int | None
    bbox: tuple[int, int, int, int] | None
    start_ms: int | None
    end_ms: int | None
    selected_count: int
    masked_pixel_count: int


@dataclass(frozen=True)
class ManifestIndex:
    ordered_entries: tuple[str, ...]
    metadata_by_entry: dict[str, EntryMetadata]
    masked_entries: set[str]


@dataclass(frozen=True)
class FrameGroup:
    key: str
    start_index: int
    end_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare aligned images from multiple directories in a simple GUI. "
            "Each panel is passed as --panel label=directory."
        ),
    )
    parser.add_argument(
        "--panel",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help=(
            "Panel definition. Example: --panel original=out/original "
            "--panel masked=out/masked"
        ),
    )
    parser.add_argument(
        "--intersection-only",
        action="store_true",
        help="Show only files that exist in every panel directory.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help=(
            "Optional relative path to open first. "
            "Example: images/000123.png or 000123.png"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional path to manifest.jsonl. "
            "If omitted, the viewer auto-detects a shared parent manifest for the masking toggle."
        ),
    )
    return parser.parse_args()


def parse_panel_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"invalid panel spec: {raw!r}. Expected LABEL=DIR.")
    label, directory = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"invalid panel spec: {raw!r}. Empty label.")
    path = Path(directory).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"panel directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"panel path is not a directory: {path}")
    return label, path


def scan_image_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = path
    return files


def build_panel_specs(raw_specs: list[str]) -> list[PanelSpec]:
    seen_labels: set[str] = set()
    panels: list[PanelSpec] = []

    for raw_spec in raw_specs:
        label, root = parse_panel_spec(raw_spec)
        if label in seen_labels:
            raise ValueError(f"duplicate panel label: {label}")
        seen_labels.add(label)

        files = scan_image_files(root)
        if not files:
            raise RuntimeError(f"no supported images found in: {root}")
        panels.append(PanelSpec(label=label, root=root, files=files))

    return panels


def build_entry_order(
    panels: list[PanelSpec],
    intersection_only: bool,
    allowed_entries: set[str] | None = None,
    manifest_order: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    file_sets = [set(panel.files) for panel in panels]
    if not file_sets:
        return []

    if intersection_only:
        entries = set.intersection(*file_sets)
    else:
        entries = set.union(*file_sets)
    if allowed_entries is not None:
        entries &= allowed_entries

    if manifest_order is None:
        return sorted(entries)

    ordered: list[str] = []
    seen: set[str] = set()
    for entry in manifest_order:
        if entry in entries and entry not in seen:
            ordered.append(entry)
            seen.add(entry)
    for entry in sorted(entries):
        if entry not in seen:
            ordered.append(entry)
    return ordered


def resolve_manifest_path(panels: list[PanelSpec], manifest_path: Path | None) -> Path | None:
    if manifest_path is not None:
        path = manifest_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"manifest path is not a file: {path}")
        return path

    parent_roots = {panel.root.parent for panel in panels}
    if len(parent_roots) != 1:
        return None

    path = next(iter(parent_roots)) / "manifest.jsonl"
    if not path.exists():
        return None
    return path


def load_manifest_index(manifest_path: Path) -> ManifestIndex:
    metadata_by_entry: dict[str, EntryMetadata] = {}
    ordered_entries: list[str] = []
    directly_masked_entries: set[str] = set()
    masked_frame_indices: set[int] = set()

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        relative_path = _manifest_entry_name(row)
        if not relative_path:
            continue

        metadata = EntryMetadata(
            relative_path=relative_path,
            frame_index=_safe_int(row.get("frame_index")),
            segment_index=_safe_int(row.get("segment_index")),
            window_id=_safe_int(row.get("window_id")),
            bbox=_safe_bbox(row.get("bbox")),
            start_ms=_safe_int(row.get("start_ms")),
            end_ms=_safe_int(row.get("end_ms")),
            selected_count=int(row.get("selected_count", 0)),
            masked_pixel_count=int(row.get("masked_pixel_count", 0)),
        )
        metadata_by_entry[relative_path] = metadata
        ordered_entries.append(relative_path)

        if metadata.selected_count > 0 or metadata.masked_pixel_count > 0:
            directly_masked_entries.add(relative_path)
            if metadata.frame_index is not None:
                masked_frame_indices.add(metadata.frame_index)

    masked_entries = set(directly_masked_entries)
    if masked_frame_indices:
        for relative_path, metadata in metadata_by_entry.items():
            if metadata.frame_index in masked_frame_indices:
                masked_entries.add(relative_path)

    return ManifestIndex(
        ordered_entries=tuple(ordered_entries),
        metadata_by_entry=metadata_by_entry,
        masked_entries=masked_entries,
    )


def load_masked_entries(manifest_path: Path) -> set[str]:
    return load_manifest_index(manifest_path).masked_entries


def _manifest_entry_name(row: dict[str, object]) -> str | None:
    for key in ("original", "masked", "mask", "lines"):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        parts = Path(value).parts
        if len(parts) >= 2:
            return Path(*parts[1:]).as_posix()
        return Path(value).as_posix()
    return None


def _safe_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _safe_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    items: list[int] = []
    for item in value:
        parsed = _safe_int(item)
        if parsed is None:
            return None
        items.append(parsed)
    return (items[0], items[1], items[2], items[3])


def _entry_numeric_prefix(entry: str) -> int | None:
    stem = Path(entry).stem
    digits: list[str] = []
    for char in stem:
        if char.isdigit():
            digits.append(char)
            continue
        if digits:
            break
    if not digits:
        return None
    return int("".join(digits))


def find_entry_index(
    entries: list[str],
    query: str,
    metadata_by_entry: dict[str, EntryMetadata] | None = None,
) -> int:
    raw = query.strip()
    if not raw:
        raise ValueError("empty segment number")

    if raw in entries:
        return entries.index(raw)

    raw_path = Path(raw)
    exact_names = [raw_path.name]
    if raw_path.suffix:
        exact_names.append(raw_path.stem)
    elif raw:
        exact_names.append(f"{raw}.png")

    for candidate in exact_names:
        for index, entry in enumerate(entries):
            if Path(entry).name == candidate:
                return index

    numeric_part = raw_path.stem if raw_path.suffix else raw
    if numeric_part.isdigit():
        segment_number = int(numeric_part)
        if metadata_by_entry is not None:
            for index, entry in enumerate(entries):
                metadata = metadata_by_entry.get(entry)
                if metadata is not None and metadata.frame_index == segment_number:
                    return index
        for index, entry in enumerate(entries):
            prefix = _entry_numeric_prefix(entry)
            if prefix is not None and prefix == segment_number:
                return index

    raise ValueError(f"segment not found: {raw}")


class ImagePanel:
    def __init__(self, parent: ttk.Frame, label: str, refresh_callback) -> None:
        self.container = ttk.Frame(parent, padding=(6, 6, 6, 6))
        self.container.columnconfigure(0, weight=1)
        self.container.rowconfigure(1, weight=1)

        self.title_var = tk.StringVar(value=label)
        self.subtitle_var = tk.StringVar(value="")
        self._label = label

        ttk.Label(
            self.container,
            textvariable=self.title_var,
            anchor="center",
            style="PanelTitle.TLabel",
        ).grid(row=0, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            self.container,
            background=BACKGROUND_COLOR,
            highlightthickness=0,
            relief="flat",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(4, 4))
        self.canvas.bind("<Configure>", lambda _event: refresh_callback())

        ttk.Label(
            self.container,
            textvariable=self.subtitle_var,
            anchor="center",
        ).grid(row=2, column=0, sticky="ew")

        self._current_source: Path | None = None
        self._source_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None

    def grid(self, *, row: int, column: int) -> None:
        self.container.grid(row=row, column=column, sticky="nsew")

    def show(self, relative_path: str, source_path: Path | None, zoom_multiplier: float) -> None:
        self.canvas.delete("all")
        self.title_var.set(self._label)

        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)

        if source_path is None:
            self.subtitle_var.set("missing")
            self._draw_message(
                width=width,
                height=height,
                text=f"Missing\n{relative_path}",
                fill=MISSING_COLOR,
            )
            self._photo = None
            return

        if self._current_source != source_path:
            self._source_image = self._load_image(source_path)
            self._current_source = source_path

        assert self._source_image is not None

        self.subtitle_var.set(source_path.name)
        image = self._source_image
        fit_scale = min(width / image.width, height / image.height)
        scale = max(0.01, fit_scale * zoom_multiplier)
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))

        if scale >= 1.0:
            resample = Image.Resampling.NEAREST
        else:
            resample = Image.Resampling.LANCZOS
        resized = image.resize((new_width, new_height), resample=resample)

        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(
            width // 2,
            height // 2,
            image=self._photo,
            anchor="center",
        )

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGBA")

    def _draw_message(self, *, width: int, height: int, text: str, fill: str) -> None:
        self.canvas.create_text(
            width // 2,
            height // 2,
            text=text,
            fill=fill,
            justify="center",
            font=("TkDefaultFont", MESSAGE_FONT_SIZE),
        )


class ComparisonApp:
    def __init__(
        self,
        root: tk.Tk,
        panels: list[PanelSpec],
        entries: list[str],
        masked_entries: set[str] | None = None,
        metadata_by_entry: dict[str, EntryMetadata] | None = None,
        start_at: str | None = None,
    ) -> None:
        self.root = root
        self.panels = panels
        self.all_entries = list(entries)
        self.masked_entries = masked_entries
        self.metadata_by_entry = metadata_by_entry or {}
        self.entries = list(entries)
        self.index = 0
        self.zoom_multiplier = 1.0
        self._refresh_job: str | None = None
        self.masking_only_var = tk.BooleanVar(value=False)
        self.frame_groups: list[FrameGroup] = []
        self._frame_group_index_by_entry: list[int] = []

        if start_at is not None:
            try:
                self.index = self.entries.index(start_at)
            except ValueError:
                pass

        self.root.title("Image Compare GUI")
        self.root.minsize(900, 520)
        self.root.geometry("1440x820")

        self.status_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="")
        self.meta_var = tk.StringVar(value="")
        self.zoom_var = tk.StringVar(value="")
        self.segment_var = tk.StringVar(value="")

        self._rebuild_frame_groups()
        self._build_widgets()
        self._bind_keys()
        self.schedule_refresh()

    def _configure_styles(self) -> None:
        default_font = ("TkDefaultFont", BASE_FONT_SIZE)
        bold_font = ("TkDefaultFont", TITLE_FONT_SIZE, "bold")
        self.root.option_add("*Font", default_font)

        style = ttk.Style(self.root)
        style.configure("TButton", padding=BUTTON_PADDING, font=default_font)
        style.configure("TLabel", font=default_font)
        style.configure("PanelTitle.TLabel", font=bold_font)
        style.configure("Help.TLabel", font=("TkDefaultFont", HELP_FONT_SIZE))

    def _build_widgets(self) -> None:
        self._configure_styles()

        outer = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(12, weight=1)

        ttk.Button(controls, text="Prev", command=self.show_previous).grid(
            row=0,
            column=0,
            padx=(0, 6),
        )
        ttk.Button(controls, text="Next", command=self.show_next).grid(
            row=0,
            column=1,
            padx=(0, 6),
        )
        ttk.Button(controls, text="Prev Frame", command=self.show_previous_frame).grid(
            row=0,
            column=2,
            padx=(0, 6),
        )
        ttk.Button(controls, text="Next Frame", command=self.show_next_frame).grid(
            row=0,
            column=3,
            padx=(0, 12),
        )
        ttk.Button(controls, text="Zoom -", command=self.zoom_out).grid(
            row=0,
            column=4,
            padx=(0, 6),
        )
        ttk.Button(controls, text="Zoom +", command=self.zoom_in).grid(
            row=0,
            column=5,
            padx=(0, 6),
        )
        ttk.Button(controls, text="Fit", command=self.reset_zoom).grid(
            row=0,
            column=6,
            padx=(0, 12),
        )
        toggle = ttk.Checkbutton(
            controls,
            text="Masking Only",
            variable=self.masking_only_var,
            command=self.toggle_masking_only,
        )
        if self.masked_entries is None:
            toggle.state(["disabled"])
        toggle.grid(
            row=0,
            column=7,
            padx=(0, 12),
        )
        ttk.Label(controls, text="Segment").grid(
            row=0,
            column=8,
            padx=(0, 6),
        )
        jump_entry = ttk.Entry(
            controls,
            textvariable=self.segment_var,
            width=10,
        )
        jump_entry.grid(
            row=0,
            column=9,
            padx=(0, 6),
        )
        jump_entry.bind("<Return>", lambda _event: self.jump_to_segment())
        ttk.Button(controls, text="Jump", command=self.jump_to_segment).grid(
            row=0,
            column=10,
            padx=(0, 12),
        )
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=0,
            column=11,
            sticky="w",
        )
        ttk.Label(controls, textvariable=self.path_var).grid(
            row=0,
            column=12,
            sticky="ew",
        )
        ttk.Label(controls, textvariable=self.meta_var).grid(
            row=1,
            column=0,
            columnspan=13,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Label(controls, textvariable=self.zoom_var).grid(
            row=0,
            column=13,
            sticky="e",
        )

        content = ttk.Frame(outer)
        content.grid(row=1, column=0, sticky="nsew")
        for index in range(len(self.panels)):
            content.columnconfigure(index, weight=1, uniform="panel")
        content.rowconfigure(0, weight=1)

        self.image_panels: list[ImagePanel] = []
        for index, panel_spec in enumerate(self.panels):
            panel = ImagePanel(content, panel_spec.label, self.schedule_refresh)
            panel.grid(row=0, column=index)
            self.image_panels.append(panel)

        help_text = (
            "Keys: Left/Right or A/D navigate, PageUp/PageDown change frame, +/- zoom, 0 fit, Home/End jump, Q quit"
        )
        ttk.Label(outer, text=help_text, style="Help.TLabel").grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _event: self.show_previous())
        self.root.bind("<Right>", lambda _event: self.show_next())
        self.root.bind("<a>", lambda _event: self.show_previous())
        self.root.bind("<d>", lambda _event: self.show_next())
        self.root.bind("<Home>", lambda _event: self.go_to_index(0))
        self.root.bind("<End>", lambda _event: self.go_to_index(len(self.entries) - 1))
        self.root.bind("<Prior>", lambda _event: self.show_previous_frame())
        self.root.bind("<Next>", lambda _event: self.show_next_frame())
        self.root.bind("<minus>", lambda _event: self.zoom_out())
        self.root.bind("<underscore>", lambda _event: self.zoom_out())
        self.root.bind("<plus>", lambda _event: self.zoom_in())
        self.root.bind("<equal>", lambda _event: self.zoom_in())
        self.root.bind("<KP_Add>", lambda _event: self.zoom_in())
        self.root.bind("<KP_Subtract>", lambda _event: self.zoom_out())
        self.root.bind("<Key-0>", lambda _event: self.reset_zoom())
        self.root.bind("<q>", lambda _event: self.root.destroy())

    def schedule_refresh(self) -> None:
        if self._refresh_job is not None:
            self.root.after_cancel(self._refresh_job)
        self._refresh_job = self.root.after(20, self.refresh)

    def refresh(self) -> None:
        self._refresh_job = None
        if not self.entries:
            return

        relative_path = self.entries[self.index]
        mode = "masking-only" if self.masking_only_var.get() else "all"
        frame_group = self.frame_groups[self._frame_group_index_by_entry[self.index]]
        self.status_var.set(
            f"{self.index + 1}/{len(self.entries)} [{mode}] "
            f"frame {self._frame_group_index_by_entry[self.index] + 1}/{len(self.frame_groups)}"
        )
        self.path_var.set(relative_path)
        self.meta_var.set(self._format_entry_metadata(relative_path, frame_group))
        self.zoom_var.set(f"fit x {self.zoom_multiplier:.2f}")

        for panel_spec, panel in zip(self.panels, self.image_panels):
            panel.show(
                relative_path=relative_path,
                source_path=panel_spec.files.get(relative_path),
                zoom_multiplier=self.zoom_multiplier,
            )

    def go_to_index(self, index: int) -> None:
        if not self.entries:
            return
        self.index = max(0, min(index, len(self.entries) - 1))
        self.schedule_refresh()

    def show_previous(self) -> None:
        self.go_to_index(self.index - 1)

    def show_next(self) -> None:
        self.go_to_index(self.index + 1)

    def show_previous_frame(self) -> None:
        if not self.entries:
            return
        current_group_index = self._frame_group_index_by_entry[self.index]
        if current_group_index <= 0:
            self.go_to_index(0)
            return
        self.go_to_index(self.frame_groups[current_group_index - 1].start_index)

    def show_next_frame(self) -> None:
        if not self.entries:
            return
        current_group_index = self._frame_group_index_by_entry[self.index]
        if current_group_index + 1 >= len(self.frame_groups):
            self.go_to_index(len(self.entries) - 1)
            return
        self.go_to_index(self.frame_groups[current_group_index + 1].start_index)

    def zoom_in(self) -> None:
        self.zoom_multiplier = min(self.zoom_multiplier * ZOOM_STEP, MAX_ZOOM_MULTIPLIER)
        self.schedule_refresh()

    def zoom_out(self) -> None:
        self.zoom_multiplier = max(self.zoom_multiplier / ZOOM_STEP, MIN_ZOOM_MULTIPLIER)
        self.schedule_refresh()

    def reset_zoom(self) -> None:
        self.zoom_multiplier = 1.0
        self.schedule_refresh()

    def toggle_masking_only(self) -> None:
        current_path = self.entries[self.index] if self.entries else None

        if self.masking_only_var.get() and self.masked_entries is not None:
            filtered_entries = [
                entry
                for entry in self.all_entries
                if entry in self.masked_entries
            ]
            if not filtered_entries:
                self.masking_only_var.set(False)
                messagebox.showinfo("Image Compare GUI", "No masked entries found in manifest.")
                self.entries = list(self.all_entries)
            else:
                self.entries = filtered_entries
        else:
            self.entries = list(self.all_entries)

        self._rebuild_frame_groups()
        if not self.entries:
            self.index = 0
            self.schedule_refresh()
            return

        if current_path in self.entries:
            self.index = self.entries.index(current_path)
        else:
            self.index = min(self.index, len(self.entries) - 1)
        self.schedule_refresh()

    def jump_to_segment(self) -> None:
        try:
            index = find_entry_index(
                self.entries,
                self.segment_var.get(),
                metadata_by_entry=self.metadata_by_entry,
            )
        except ValueError as exc:
            messagebox.showinfo("Image Compare GUI", str(exc))
            return
        self.go_to_index(index)

    def _rebuild_frame_groups(self) -> None:
        self.frame_groups = []
        self._frame_group_index_by_entry = []
        if not self.entries:
            return

        current_key: str | None = None
        current_start = 0
        current_group_index = -1
        for index, entry in enumerate(self.entries):
            key = self._frame_key(entry)
            if key != current_key:
                if current_key is not None:
                    self.frame_groups.append(
                        FrameGroup(
                            key=current_key,
                            start_index=current_start,
                            end_index=index,
                        )
                    )
                current_key = key
                current_start = index
                current_group_index += 1
            self._frame_group_index_by_entry.append(current_group_index)

        assert current_key is not None
        self.frame_groups.append(
            FrameGroup(
                key=current_key,
                start_index=current_start,
                end_index=len(self.entries),
            )
        )

    def _frame_key(self, entry: str) -> str:
        metadata = self.metadata_by_entry.get(entry)
        if metadata is not None and metadata.frame_index is not None:
            return f"frame:{metadata.frame_index}"
        prefix = _entry_numeric_prefix(entry)
        if prefix is not None:
            return f"prefix:{prefix}"
        return f"path:{entry}"

    def _format_entry_metadata(self, entry: str, frame_group: FrameGroup) -> str:
        metadata = self.metadata_by_entry.get(entry)
        if metadata is None:
            return f"{frame_group.key}  members {frame_group.end_index - frame_group.start_index}"

        parts = []
        if metadata.frame_index is not None:
            parts.append(f"frame {metadata.frame_index}")
        if metadata.segment_index is not None:
            parts.append(f"segment {metadata.segment_index}")
        if metadata.window_id is not None:
            parts.append(f"window {metadata.window_id}")
        if metadata.bbox is not None:
            left, top, right, bottom = metadata.bbox
            parts.append(f"bbox {left},{top},{right},{bottom}")
        if metadata.start_ms is not None and metadata.end_ms is not None:
            parts.append(f"time {metadata.start_ms}-{metadata.end_ms} ms")
        if metadata.masked_pixel_count > 0:
            parts.append(f"masked_px {metadata.masked_pixel_count}")
        return "  |  ".join(parts)


def main() -> int:
    args = parse_args()

    try:
        panels = build_panel_specs(args.panel)
        manifest_path = resolve_manifest_path(panels, args.manifest)
        manifest_index = load_manifest_index(manifest_path) if manifest_path is not None else None
        masked_entries = manifest_index.masked_entries if manifest_index is not None else None
        entries = build_entry_order(
            panels,
            intersection_only=args.intersection_only,
            manifest_order=manifest_index.ordered_entries if manifest_index is not None else None,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    if not entries:
        raise SystemExit("no aligned images found to compare")

    root = tk.Tk()
    try:
        app = ComparisonApp(
            root,
            panels,
            entries,
            masked_entries=masked_entries,
            metadata_by_entry=manifest_index.metadata_by_entry if manifest_index is not None else None,
            start_at=args.start_at,
        )
        app.refresh()
        root.mainloop()
        return 0
    except Exception as exc:
        messagebox.showerror("Image Compare GUI", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
