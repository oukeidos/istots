from __future__ import annotations

import hashlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from PIL import Image

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
_THRESHOLD = 245
_EROSION_RADIUS = 1
_MIN_COMPONENT_AREA = 4
_VISIBLE_THRESHOLD = 255
_SCANLINE_MATCH_THRESHOLD = 0.6
_REPRESENTATIVE_OWNER_RATIO = 0.2
_MICROBAND_RATIO = 0.22
_MERGE_GAP_RATIO = 0.12
_SIGNATURE_OVERLAP_THRESHOLD = 0.7
_THICKNESS_SIMILARITY_THRESHOLD = 0.6
_CORE_CLUSTER_GAP = 12.0
_CORE_CLUSTER_STEPS = 8
_FURIGANA_FRAGMENT_GAP_RATIO = 0.85
_FURIGANA_FRAGMENT_MIN_OVERLAP = 0.55
_PARALLEL_ANALYZE_MIN_IMAGES = 256
_PARALLEL_ANALYZE_MAX_WORKERS = 8
_PARALLEL_ANALYZE_TARGET_CHUNKS_PER_WORKER = 8
LineRole = Literal["main", "furigana", "other"]


@dataclass(frozen=True)
class Component:
    left: int
    top: int
    right: int
    bottom: int
    area: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    def thickness(self, orientation: str) -> int:
        return self.height if orientation == HORIZONTAL else self.width

    def cross_start(self, orientation: str) -> int:
        return self.top if orientation == HORIZONTAL else self.left

    def cross_end(self, orientation: str) -> int:
        return self.bottom if orientation == HORIZONTAL else self.right

    def cross_center(self, orientation: str) -> float:
        start = self.cross_start(orientation)
        end = self.cross_end(orientation)
        return (start + end) / 2.0


@dataclass(frozen=True)
class LineCluster:
    component_indices: tuple[int, ...]
    support_indices: tuple[int, ...]
    left: int
    top: int
    right: int
    bottom: int
    content_left: int
    content_top: int
    content_right: int
    content_bottom: int
    area: int
    representative_thickness: float

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    def thickness(self, orientation: str) -> int:
        return self.height if orientation == HORIZONTAL else self.width

    def flow_start(self, orientation: str) -> int:
        return self.content_left if orientation == HORIZONTAL else self.content_top

    def flow_end(self, orientation: str) -> int:
        return self.content_right if orientation == HORIZONTAL else self.content_bottom


@dataclass(frozen=True)
class _ScanlineSlab:
    start: int
    end: int
    owners: frozenset[int]


@dataclass(frozen=True)
class _ScanlineBand:
    start: int
    end: int
    slab_start: int
    slab_end: int


@dataclass(frozen=True)
class FuriganaMaskResult:
    image: Image.Image
    mask: Image.Image
    orientation: str | None
    component_count: int
    selected_count: int
    masked_pixel_count: int
    lines: tuple["LineDebugInfo", ...]


@dataclass(frozen=True)
class LineDebugInfo:
    left: int
    top: int
    right: int
    bottom: int
    role: LineRole


@dataclass(frozen=True)
class OrientationLineStats:
    main_thickness: float
    main_min_thickness: float
    furigana_min_thickness: float
    furigana_max_thickness: float
    max_cross_gap: float
    min_flow_overlap_ratio: float


@dataclass(frozen=True)
class GlobalLineStats:
    horizontal: OrientationLineStats | None = None
    vertical: OrientationLineStats | None = None

    def for_orientation(self, orientation: str | None) -> OrientationLineStats | None:
        if orientation == HORIZONTAL:
            return self.horizontal
        if orientation == VERTICAL:
            return self.vertical
        return None


@dataclass
class _FrameAnalysis:
    image: Image.Image
    ink_mask: np.ndarray
    core_mask: np.ndarray
    labels: np.ndarray
    components: tuple[Component, ...]
    orientation: str | None
    lines: tuple[LineCluster, ...]


def apply_furigana_mask(image: Image.Image) -> Image.Image:
    return build_furigana_mask(image).image


def build_furigana_mask(
    image: Image.Image,
    global_stats: GlobalLineStats | None = None,
) -> FuriganaMaskResult:
    analysis = _analyze_image(image)
    if global_stats is None:
        global_stats = _build_global_line_stats((analysis,))
    return _finalize_analysis(analysis, global_stats)


def build_furigana_masks(images: Sequence[Image.Image]) -> list[FuriganaMaskResult]:
    analyses = _analyze_images(images)
    global_stats = _build_global_line_stats(analyses)
    return [_finalize_analysis(analysis, global_stats) for analysis in analyses]


def _analyze_images(images: Sequence[Image.Image]) -> list[_FrameAnalysis]:
    unique_images, analysis_order = _deduplicate_analysis_images(images)
    worker_count = _parallel_analyze_worker_count(len(unique_images))
    if worker_count < 2:
        unique_analyses = [_analyze_image(image) for image in unique_images]
    else:
        target_chunks_per_worker = _parallel_analyze_target_chunks_per_worker()
        chunk_size = max(1, len(unique_images) // (worker_count * target_chunks_per_worker))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=_parallel_analyze_context(),
        ) as executor:
            unique_analyses = list(executor.map(_analyze_image, unique_images, chunksize=chunk_size))

    return [unique_analyses[index] for index in analysis_order]


def _deduplicate_analysis_images(images: Sequence[Image.Image]) -> tuple[list[Image.Image], list[int]]:
    unique_images: list[Image.Image] = []
    analysis_order: list[int] = []
    digest_buckets: dict[bytes, list[int]] = {}

    for image in images:
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        image_bytes = rgb_image.tobytes()
        digest = hashlib.blake2b(image_bytes, digest_size=16).digest()
        match_index: int | None = None
        bucket = digest_buckets.get(digest)
        if bucket is not None:
            for candidate_index in bucket:
                candidate_image = unique_images[candidate_index]
                if candidate_image.size != rgb_image.size:
                    continue
                if candidate_image.tobytes() == image_bytes:
                    match_index = candidate_index
                    break
        if match_index is None:
            match_index = len(unique_images)
            unique_images.append(rgb_image)
            digest_buckets.setdefault(digest, []).append(match_index)
        analysis_order.append(match_index)

    return unique_images, analysis_order


def _parallel_analyze_worker_count(image_count: int) -> int:
    if image_count < _parallel_analyze_min_images():
        return 1
    if multiprocessing.current_process().name != "MainProcess":
        return 1

    cpu_count = os.cpu_count() or 1
    if cpu_count < 2:
        return 1
    return min(_parallel_analyze_max_workers(), cpu_count, image_count)


def _parallel_analyze_min_images() -> int:
    return _parallel_env_int("ISTOTS_FURIGANA_MASK_PARALLEL_MIN_IMAGES", _PARALLEL_ANALYZE_MIN_IMAGES)


def _parallel_analyze_max_workers() -> int:
    return _parallel_env_int("ISTOTS_FURIGANA_MASK_PARALLEL_MAX_WORKERS", _PARALLEL_ANALYZE_MAX_WORKERS)


def _parallel_analyze_target_chunks_per_worker() -> int:
    return _parallel_env_int(
        "ISTOTS_FURIGANA_MASK_PARALLEL_TARGET_CHUNKS_PER_WORKER",
        _PARALLEL_ANALYZE_TARGET_CHUNKS_PER_WORKER,
    )


def _parallel_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed >= 1 else default


def _parallel_analyze_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _analyze_image(image: Image.Image) -> _FrameAnalysis:
    rgb_image = image.convert("RGB")
    gray_array = np.asarray(rgb_image.convert("L"), dtype=np.uint8)
    ink_mask = gray_array < _VISIBLE_THRESHOLD

    if not np.any(ink_mask):
        return _FrameAnalysis(
            image=rgb_image,
            ink_mask=ink_mask,
            core_mask=ink_mask,
            labels=np.zeros_like(ink_mask, dtype=np.int32),
            components=(),
            orientation=None,
            lines=(),
        )

    core_mask = _build_core_mask(gray_array, ink_mask)
    components, labels = _extract_components(core_mask, erode=False)
    if len(components) < 2:
        fallback_components, fallback_labels = _extract_components(ink_mask, erode=True)
        if len(fallback_components) > len(components):
            core_mask = ink_mask
            components = fallback_components
            labels = fallback_labels
    if len(components) < 2:
        return _FrameAnalysis(
            image=rgb_image,
            ink_mask=ink_mask,
            core_mask=core_mask,
            labels=labels,
            components=tuple(components),
            orientation=None,
            lines=(),
        )

    orientation = _infer_orientation(components, image_size=(rgb_image.width, rgb_image.height))
    lines = _build_lines(components, core_mask.shape, orientation)
    return _FrameAnalysis(
        image=rgb_image,
        ink_mask=ink_mask,
        core_mask=core_mask,
        labels=labels,
        components=tuple(components),
        orientation=orientation,
        lines=tuple(lines),
    )


def _finalize_analysis(
    analysis: _FrameAnalysis,
    global_stats: GlobalLineStats,
) -> FuriganaMaskResult:
    if analysis.orientation is None or len(analysis.components) < 2:
        empty_mask = np.zeros_like(analysis.ink_mask, dtype=bool)
        return FuriganaMaskResult(
            image=analysis.image.copy(),
            mask=_mask_to_image(empty_mask),
            orientation=analysis.orientation,
            component_count=len(analysis.components),
            selected_count=0,
            masked_pixel_count=0,
            lines=(),
        )

    stats = global_stats.for_orientation(analysis.orientation)
    if stats is None:
        stats = _build_orientation_stats(analysis.lines)
    lines = _coalesce_lines(list(analysis.lines), analysis.orientation, stats)
    display_boxes = _expand_line_boxes(lines, analysis.ink_mask, analysis.orientation)

    selected_line_indices, line_roles = _select_furigana_components(
        lines=lines,
        orientation=analysis.orientation,
        stats=stats,
    )
    selected_line_indices, line_roles = _attach_furigana_fragments(
        lines=lines,
        selected_line_indices=selected_line_indices,
        line_roles=line_roles,
        orientation=analysis.orientation,
        stats=stats,
    )
    selected_mask = _build_selected_mask(
        selected_line_indices=selected_line_indices,
        boxes=display_boxes,
        mask_shape=analysis.ink_mask.shape,
    )
    debug_lines = tuple(
        LineDebugInfo(
            left=box[0],
            top=box[1],
            right=box[2],
            bottom=box[3],
            role=line_roles[index],
        )
        for index, box in enumerate(display_boxes)
    )
    selected_component_indices = {
        component_index
        for line_index in selected_line_indices
        for component_index in lines[line_index].component_indices
    }

    if not np.any(selected_mask):
        return FuriganaMaskResult(
            image=analysis.image.copy(),
            mask=_mask_to_image(selected_mask),
            orientation=analysis.orientation,
            component_count=len(analysis.components),
            selected_count=0,
            masked_pixel_count=0,
            lines=debug_lines,
        )

    masked_array = np.asarray(analysis.image, dtype=np.uint8).copy()
    masked_array[selected_mask] = 255
    return FuriganaMaskResult(
        image=Image.fromarray(masked_array, mode="RGB"),
        mask=_mask_to_image(selected_mask),
        orientation=analysis.orientation,
        component_count=len(analysis.components),
        selected_count=len(selected_component_indices),
        masked_pixel_count=int(selected_mask.sum()),
        lines=debug_lines,
    )


def _coalesce_lines(
    lines: list[LineCluster],
    orientation: str,
    stats: OrientationLineStats | None,
) -> list[LineCluster]:
    if len(lines) < 2 or stats is None:
        return lines

    max_combined_thickness = max(stats.main_min_thickness, stats.main_thickness * 1.15)
    merged: list[LineCluster] = []
    current = lines[0]

    for line in lines[1:]:
        gap = _cross_gap(current, line, orientation)
        combined_thickness = current.representative_thickness + gap + line.representative_thickness
        overlap_ratio = _flow_overlap_ratio(current, line, orientation)
        if gap <= 1 and overlap_ratio >= stats.min_flow_overlap_ratio and combined_thickness <= max_combined_thickness:
            current = _merge_line_clusters(current, line, orientation)
            continue

        merged.append(current)
        current = line

    merged.append(current)
    return merged


def _merge_line_clusters(
    left: LineCluster,
    right: LineCluster,
    orientation: str,
) -> LineCluster:
    if orientation == HORIZONTAL:
        merged_left = min(left.left, right.left)
        merged_right = max(left.right, right.right)
        merged_top = min(left.top, right.top)
        merged_bottom = max(left.bottom, right.bottom)
    else:
        merged_left = min(left.left, right.left)
        merged_right = max(left.right, right.right)
        merged_top = min(left.top, right.top)
        merged_bottom = max(left.bottom, right.bottom)

    return LineCluster(
        component_indices=tuple(sorted(set(left.component_indices) | set(right.component_indices))),
        support_indices=tuple(sorted(set(left.support_indices) | set(right.support_indices))),
        left=merged_left,
        top=merged_top,
        right=merged_right,
        bottom=merged_bottom,
        content_left=min(left.content_left, right.content_left),
        content_top=min(left.content_top, right.content_top),
        content_right=max(left.content_right, right.content_right),
        content_bottom=max(left.content_bottom, right.content_bottom),
        area=left.area + right.area,
        representative_thickness=float(
            (merged_bottom - merged_top + 1) if orientation == HORIZONTAL else (merged_right - merged_left + 1)
        ),
    )


def _cross_gap(
    left: LineCluster,
    right: LineCluster,
    orientation: str,
) -> int:
    if orientation == HORIZONTAL:
        return _line_gap(left.top, left.bottom, right.top, right.bottom)
    return _line_gap(left.left, left.right, right.left, right.right)


def _build_global_line_stats(analyses: Sequence[_FrameAnalysis]) -> GlobalLineStats:
    horizontal_lines = [
        line
        for analysis in analyses
        if analysis.orientation == HORIZONTAL
        for line in analysis.lines
    ]
    vertical_lines = [
        line
        for analysis in analyses
        if analysis.orientation == VERTICAL
        for line in analysis.lines
    ]
    return GlobalLineStats(
        horizontal=_build_orientation_stats(horizontal_lines),
        vertical=_build_orientation_stats(vertical_lines),
    )


def _build_orientation_stats(lines: Sequence[LineCluster]) -> OrientationLineStats | None:
    if len(lines) < 2:
        return None

    thicknesses = np.asarray([line.representative_thickness for line in lines], dtype=np.float64)
    weights = np.asarray([line.area for line in lines], dtype=np.float64)
    pivot = _weighted_percentile(thicknesses, weights, 0.60)
    main_values = thicknesses[thicknesses >= pivot]
    main_weights = weights[thicknesses >= pivot]
    if len(main_values) == 0:
        main_values = thicknesses
        main_weights = weights

    main_thickness = _weighted_median(main_values, main_weights)
    if main_thickness <= 0:
        return None

    return OrientationLineStats(
        main_thickness=main_thickness,
        main_min_thickness=max(3.0, main_thickness * 0.84),
        furigana_min_thickness=max(2.0, main_thickness * 0.18),
        furigana_max_thickness=max(2.0, main_thickness * 0.76),
        max_cross_gap=max(2.0, main_thickness * 1.4),
        min_flow_overlap_ratio=0.20,
    )


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _build_core_mask(gray_array: np.ndarray, ink_mask: np.ndarray) -> np.ndarray:
    if not np.any(ink_mask):
        return ink_mask.copy()

    foreground_values = gray_array[ink_mask]
    threshold = _estimate_core_threshold(foreground_values)
    core_mask = ink_mask & (gray_array <= threshold)
    if np.any(core_mask):
        return core_mask
    return ink_mask.copy()


def _estimate_core_threshold(foreground_values: np.ndarray) -> int:
    values = np.asarray(foreground_values, dtype=np.uint8)
    if values.size == 0:
        return _THRESHOLD

    counts = np.bincount(values, minlength=256).astype(np.int64, copy=False)
    nonzero_levels = np.flatnonzero(counts)
    low = float(nonzero_levels[0])
    high = float(nonzero_levels[-1])
    if high - low < 1.0:
        return int(min(_VISIBLE_THRESHOLD - 1, round(high)))

    cumulative = np.cumsum(counts)
    centers = np.asarray(
        [
            _histogram_percentile(cumulative, 15.0),
            _histogram_percentile(cumulative, 85.0),
        ],
        dtype=np.float64,
    )
    if abs(float(centers[1] - centers[0])) < 1.0:
        centers = np.asarray([low, high], dtype=np.float64)

    weighted_levels = nonzero_levels.astype(np.float64, copy=False)
    weighted_counts = counts[nonzero_levels].astype(np.float64, copy=False)
    for _ in range(_CORE_CLUSTER_STEPS):
        dark_mask = np.abs(weighted_levels - centers[0]) <= np.abs(weighted_levels - centers[1])
        updated = centers.copy()
        if np.any(dark_mask):
            updated[0] = float(np.average(weighted_levels[dark_mask], weights=weighted_counts[dark_mask]))
        if np.any(~dark_mask):
            updated[1] = float(np.average(weighted_levels[~dark_mask], weights=weighted_counts[~dark_mask]))
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated

    dark_center, light_center = sorted(float(value) for value in centers.tolist())
    if (light_center - dark_center) < _CORE_CLUSTER_GAP:
        return int(min(_VISIBLE_THRESHOLD - 1, round(light_center)))
    return int(min(_VISIBLE_THRESHOLD - 1, np.floor((dark_center + light_center) / 2.0)))


def _histogram_percentile(cumulative_counts: np.ndarray, percentile: float) -> float:
    total = int(cumulative_counts[-1]) if cumulative_counts.size else 0
    if total <= 0:
        return float(_THRESHOLD)

    rank = (total - 1) * (percentile / 100.0)
    lower_rank = int(np.floor(rank))
    upper_rank = int(np.ceil(rank))
    lower_value = float(np.searchsorted(cumulative_counts, lower_rank + 1, side="left"))
    upper_value = float(np.searchsorted(cumulative_counts, upper_rank + 1, side="left"))
    if lower_rank == upper_rank:
        return lower_value
    return lower_value + ((upper_value - lower_value) * (rank - lower_rank))


def _extract_components(mask: np.ndarray, erode: bool) -> tuple[list[Component], np.ndarray]:
    working_mask = mask
    if erode:
        eroded = _erode_mask(mask, radius_x=_EROSION_RADIUS, radius_y=_EROSION_RADIUS)
        if np.any(eroded):
            working_mask = eroded
    return _find_components(working_mask)


def _erode_mask(mask: np.ndarray, radius_x: int, radius_y: int) -> np.ndarray:
    if radius_x <= 0 and radius_y <= 0:
        return mask.copy()
    padded = np.pad(
        mask,
        ((radius_y, radius_y), (radius_x, radius_x)),
        mode="constant",
        constant_values=False,
    )
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range((radius_y * 2) + 1):
        for dx in range((radius_x * 2) + 1):
            eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return eroded


def _find_components(mask: np.ndarray) -> tuple[list[Component], np.ndarray]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if height == 0 or width == 0 or not np.any(mask):
        return [], labels

    run_rows, run_starts, run_ends, row_run_offsets = _foreground_runs(mask)
    run_count = int(run_rows.shape[0])
    if run_count == 0:
        return [], labels

    parents = np.arange(run_count, dtype=np.int32)
    ranks = np.zeros(run_count, dtype=np.uint8)

    for row_index in range(1, height):
        prev_start = int(row_run_offsets[row_index - 1])
        prev_end = int(row_run_offsets[row_index])
        curr_start = int(row_run_offsets[row_index])
        curr_end = int(row_run_offsets[row_index + 1])
        if prev_start == prev_end or curr_start == curr_end:
            continue

        prev_pointer = prev_start
        for curr_index in range(curr_start, curr_end):
            curr_left = int(run_starts[curr_index])
            curr_right = int(run_ends[curr_index])
            while prev_pointer < prev_end and int(run_ends[prev_pointer]) + 1 < curr_left:
                prev_pointer += 1

            scan_index = prev_pointer
            while scan_index < prev_end and int(run_starts[scan_index]) <= curr_right + 1:
                _union_runs_array(parents, ranks, curr_index, int(scan_index))
                scan_index += 1

    run_roots = np.fromiter((_find_run_root_array(parents, run_index) for run_index in range(run_count)), dtype=np.int32)
    unique_roots, first_indices, inverse = np.unique(
        run_roots,
        return_index=True,
        return_inverse=True,
    )
    root_order = np.argsort(first_indices, kind="stable")
    ordered_roots = unique_roots[root_order]
    component_count = int(ordered_roots.shape[0])

    remap = np.empty(component_count, dtype=np.int32)
    remap[root_order] = np.arange(component_count, dtype=np.int32)
    component_indices = remap[inverse]

    component_lefts = np.full(component_count, width, dtype=np.int32)
    component_rights = np.zeros(component_count, dtype=np.int32)
    component_tops = np.full(component_count, height, dtype=np.int32)
    component_bottoms = np.zeros(component_count, dtype=np.int32)
    component_areas = np.zeros(component_count, dtype=np.int32)

    np.minimum.at(component_lefts, component_indices, run_starts)
    np.maximum.at(component_rights, component_indices, run_ends)
    np.minimum.at(component_tops, component_indices, run_rows)
    np.maximum.at(component_bottoms, component_indices, run_rows)
    np.add.at(component_areas, component_indices, (run_ends - run_starts + 1).astype(np.int32, copy=False))

    keep_mask = component_areas >= _MIN_COMPONENT_AREA
    kept_indices = np.flatnonzero(keep_mask)

    components: list[Component] = [
        Component(
            left=int(component_lefts[component_index]),
            top=int(component_tops[component_index]),
            right=int(component_rights[component_index]),
            bottom=int(component_bottoms[component_index]),
            area=int(component_areas[component_index]),
        )
        for component_index in kept_indices
    ]

    component_labels = np.zeros(component_count, dtype=np.int32)
    component_labels[kept_indices] = np.arange(1, kept_indices.size + 1, dtype=np.int32)
    run_labels = component_labels[component_indices]

    for run_index, label in enumerate(run_labels):
        if label == 0:
            continue
        row = int(run_rows[run_index])
        left = int(run_starts[run_index])
        right = int(run_ends[run_index])
        labels[row, left : right + 1] = int(label)

    return components, labels


def _foreground_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = mask.shape[0]
    transitions = np.diff(
        np.pad(mask.astype(np.int8, copy=False), ((0, 0), (1, 1)), mode="constant", constant_values=0),
        axis=1,
    )
    run_rows, run_starts = np.nonzero(transitions == 1)
    end_rows, run_end_plus_one = np.nonzero(transitions == -1)
    if run_rows.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.zeros(height + 1, dtype=np.int32),
        )

    if not np.array_equal(run_rows, end_rows):
        raise RuntimeError("foreground run extraction lost row alignment")

    row_run_counts = np.bincount(run_rows, minlength=height).astype(np.int32, copy=False)
    row_run_offsets = np.empty(height + 1, dtype=np.int32)
    row_run_offsets[0] = 0
    np.cumsum(row_run_counts, out=row_run_offsets[1:])
    return (
        run_rows.astype(np.int32, copy=False),
        run_starts.astype(np.int32, copy=False),
        (run_end_plus_one - 1).astype(np.int32, copy=False),
        row_run_offsets,
    )
def _find_run_root(parents: list[int], index: int) -> int:
    root = index
    while parents[root] != root:
        root = parents[root]
    while parents[index] != index:
        parent = parents[index]
        parents[index] = root
        index = parent
    return root


def _union_runs(parents: list[int], ranks: list[int], left: int, right: int) -> None:
    left_root = _find_run_root(parents, left)
    right_root = _find_run_root(parents, right)
    if left_root == right_root:
        return

    if ranks[left_root] < ranks[right_root]:
        parents[left_root] = right_root
        return
    if ranks[left_root] > ranks[right_root]:
        parents[right_root] = left_root
        return

    parents[right_root] = left_root
    ranks[left_root] += 1


def _find_run_root_array(parents: np.ndarray, index: int) -> int:
    root = index
    while True:
        parent = int(parents[root])
        if parent == root:
            break
        root = parent
    while True:
        parent = int(parents[index])
        if parent == index:
            break
        parents[index] = root
        index = parent
    return root


def _union_runs_array(parents: np.ndarray, ranks: np.ndarray, left: int, right: int) -> None:
    left_root = _find_run_root_array(parents, left)
    right_root = _find_run_root_array(parents, right)
    if left_root == right_root:
        return

    left_rank = int(ranks[left_root])
    right_rank = int(ranks[right_root])
    if left_rank < right_rank:
        parents[left_root] = right_root
        return
    if left_rank > right_rank:
        parents[right_root] = left_root
        return

    parents[right_root] = left_root
    ranks[left_root] = left_rank + 1
def _neighbors(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            next_x = x + dx
            next_y = y + dy
            if 0 <= next_x < width and 0 <= next_y < height:
                result.append((next_x, next_y))
    return result


def _infer_orientation(components: list[Component], image_size: tuple[int, int]) -> str:
    image_width, image_height = image_size
    if image_width > image_height * 1.2:
        return HORIZONTAL
    if image_height > image_width * 1.2:
        return VERTICAL

    widths = np.asarray([component.width for component in components], dtype=np.float64)
    heights = np.asarray([component.height for component in components], dtype=np.float64)
    areas = np.asarray([component.area for component in components], dtype=np.float64)

    median_width = _weighted_median(widths, areas)
    median_height = _weighted_median(heights, areas)
    if median_height > median_width * 1.05:
        return HORIZONTAL
    if median_width > median_height * 1.05:
        return VERTICAL

    global_left = min(component.left for component in components)
    global_top = min(component.top for component in components)
    global_right = max(component.right for component in components)
    global_bottom = max(component.bottom for component in components)
    width = global_right - global_left + 1
    height = global_bottom - global_top + 1
    if width >= height:
        return HORIZONTAL
    return VERTICAL


def _build_lines(
    components: list[Component],
    mask_shape: tuple[int, int],
    orientation: str,
) -> list[LineCluster]:
    if not components:
        return []

    thicknesses = np.asarray([component.thickness(orientation) for component in components], dtype=np.float64)
    weights = np.asarray([component.area for component in components], dtype=np.float64)
    median_thickness = _weighted_median(thicknesses, weights)
    slabs = _build_scanline_owner_sets(components, mask_shape, orientation)
    bands = _build_scanline_bands(slabs)
    if not bands:
        return []

    bands = _merge_scanline_bands(
        bands=bands,
        owner_sets=slabs,
        components=components,
        orientation=orientation,
        median_thickness=median_thickness,
    )
    return [
        _make_line_cluster_from_band(
            band=band,
            owner_sets=slabs,
            components=components,
            mask_shape=mask_shape,
            orientation=orientation,
        )
        for band in bands
    ]


def _build_scanline_owner_sets(
    components: Sequence[Component],
    mask_shape: tuple[int, int],
    orientation: str,
) -> list[_ScanlineSlab]:
    axis_length = mask_shape[0] if orientation == HORIZONTAL else mask_shape[1]
    if axis_length <= 0 or not components:
        return []

    start_events: list[list[int]] = [[] for _ in range(axis_length + 1)]
    end_events: list[list[int]] = [[] for _ in range(axis_length + 1)]
    if orientation == HORIZONTAL:
        for component_index, component in enumerate(components):
            start_events[component.top].append(component_index)
            end_after = component.bottom + 1
            if end_after <= axis_length:
                end_events[end_after].append(component_index)
    else:
        for component_index, component in enumerate(components):
            start_events[component.left].append(component_index)
            end_after = component.right + 1
            if end_after <= axis_length:
                end_events[end_after].append(component_index)

    change_positions = sorted(
        {
            position
            for position in range(axis_length + 1)
            if start_events[position] or end_events[position] or position == axis_length
        }
    )

    active: set[int] = set()
    slabs: list[_ScanlineSlab] = []
    current_position = 0

    for position in change_positions:
        if position > current_position and active:
            slabs.append(
                _ScanlineSlab(
                    start=current_position,
                    end=position - 1,
                    owners=frozenset(active),
                )
            )

        for component_index in end_events[position]:
            active.discard(component_index)
        for component_index in start_events[position]:
            active.add(component_index)

        current_position = position

    return slabs


def _build_scanline_bands(owner_sets: list[_ScanlineSlab]) -> list[_ScanlineBand]:
    bands: list[_ScanlineBand] = []
    if not owner_sets:
        return bands

    current_start = owner_sets[0].start
    current_end = owner_sets[0].end
    current_slab_start = 0
    previous_owners = owner_sets[0].owners

    for slab_index in range(1, len(owner_sets)):
        slab = owner_sets[slab_index]
        gap = slab.start - current_end - 1
        if gap == 0 and _owner_set_overlap(previous_owners, slab.owners) >= _SCANLINE_MATCH_THRESHOLD:
            current_end = slab.end
            previous_owners = slab.owners
            continue

        bands.append(
            _ScanlineBand(
                start=current_start,
                end=current_end,
                slab_start=current_slab_start,
                slab_end=slab_index - 1,
            )
        )
        current_start = slab.start
        current_end = slab.end
        current_slab_start = slab_index
        previous_owners = slab.owners

    bands.append(
        _ScanlineBand(
            start=current_start,
            end=current_end,
            slab_start=current_slab_start,
            slab_end=len(owner_sets) - 1,
        )
    )
    return bands


def _merge_scanline_bands(
    bands: list[_ScanlineBand],
    owner_sets: list[_ScanlineSlab],
    components: list[Component],
    orientation: str,
    median_thickness: float,
) -> list[_ScanlineBand]:
    if len(bands) < 2:
        return bands

    merge_gap = max(0, int(round(median_thickness * _MERGE_GAP_RATIO)))
    micro_thickness = max(1, int(round(median_thickness * _MICROBAND_RATIO)))
    merged = list(bands)

    changed = True
    while changed:
        changed = False

        index = 0
        next_bands: list[tuple[int, int]] = []
        while index < len(merged):
            current = merged[index]
            if index + 1 < len(merged):
                following = merged[index + 1]
                gap = following.start - current.end - 1
                overlap = _band_overlap_signature(current, following, owner_sets)
                if gap <= merge_gap and (
                    overlap >= _SIGNATURE_OVERLAP_THRESHOLD
                    or (
                        gap == 0
                        and _band_scale_similarity(current, following, owner_sets, components, orientation)
                        >= _THICKNESS_SIMILARITY_THRESHOLD
                    )
                ):
                    next_bands.append(_combine_scanline_bands(current, following))
                    index += 2
                    changed = True
                    continue
            next_bands.append(current)
            index += 1
        merged = next_bands

        if len(merged) < 2:
            break

        index = 0
        next_bands = []
        while index < len(merged):
            current = merged[index]
            thickness = current.end - current.start + 1
            if thickness <= micro_thickness:
                previous = next_bands[-1] if next_bands else None
                following = merged[index + 1] if index + 1 < len(merged) else None
                previous_overlap = _band_overlap_signature(previous, current, owner_sets) if previous else 0.0
                following_overlap = _band_overlap_signature(current, following, owner_sets) if following else 0.0
                previous_similarity = (
                    _band_scale_similarity(previous, current, owner_sets, components, orientation)
                    if previous
                    else 0.0
                )
                following_similarity = (
                    _band_scale_similarity(current, following, owner_sets, components, orientation)
                    if following
                    else 0.0
                )

                if previous and (
                    previous_overlap >= _SIGNATURE_OVERLAP_THRESHOLD
                    or (previous.end + 1 == current.start and previous_similarity >= _THICKNESS_SIMILARITY_THRESHOLD)
                ):
                    next_bands[-1] = _combine_scanline_bands(previous, current)
                    index += 1
                    changed = True
                    continue

                if following and (
                    following_overlap >= _SIGNATURE_OVERLAP_THRESHOLD
                    or (current.end + 1 == following.start and following_similarity >= _THICKNESS_SIMILARITY_THRESHOLD)
                ):
                    next_bands.append(_combine_scanline_bands(current, following))
                    index += 2
                    changed = True
                    continue
            next_bands.append(current)
            index += 1
        merged = next_bands

    return merged


def _band_overlap_signature(
    left_band: _ScanlineBand | None,
    right_band: _ScanlineBand | None,
    owner_sets: list[_ScanlineSlab],
) -> float:
    if left_band is None or right_band is None:
        return 0.0

    left_signature = _representative_owners(
        _band_owner_counts(left_band, owner_sets),
        left_band.end - left_band.start + 1,
    )
    right_signature = _representative_owners(
        _band_owner_counts(right_band, owner_sets),
        right_band.end - right_band.start + 1,
    )
    return _owner_set_overlap(left_signature, right_signature)


def _band_scale_similarity(
    left_band: _ScanlineBand | None,
    right_band: _ScanlineBand | None,
    owner_sets: list[_ScanlineSlab],
    components: list[Component],
    orientation: str,
) -> float:
    if left_band is None or right_band is None:
        return 0.0

    del owner_sets, components, orientation
    left_scale = float((left_band.end - left_band.start) + 1)
    right_scale = float((right_band.end - right_band.start) + 1)
    if left_scale <= 0 or right_scale <= 0:
        return 0.0
    return min(left_scale, right_scale) / max(left_scale, right_scale)


def _band_owner_counts(
    band: _ScanlineBand,
    owner_sets: list[_ScanlineSlab],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for slab_index in range(band.slab_start, band.slab_end + 1):
        slab = owner_sets[slab_index]
        thickness = slab.end - slab.start + 1
        for index in slab.owners:
            counts[index] = counts.get(index, 0) + thickness
    return counts


def _representative_owners(counts: dict[int, int], thickness: int) -> frozenset[int]:
    if not counts:
        return frozenset()

    minimum_support = max(1, int(np.ceil(thickness * _REPRESENTATIVE_OWNER_RATIO)))
    owners = {index for index, count in counts.items() if count >= minimum_support}
    if owners:
        return frozenset(owners)

    peak = max(counts.values())
    return frozenset(index for index, count in counts.items() if count == peak)


def _owner_set_overlap(left: frozenset[int], right: frozenset[int]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return overlap / float(union)


def _make_line_cluster_from_band(
    band: _ScanlineBand,
    owner_sets: list[_ScanlineSlab],
    components: list[Component],
    mask_shape: tuple[int, int],
    orientation: str,
) -> LineCluster:
    all_counts = _band_owner_counts(band, owner_sets)
    all_indices = tuple(sorted(all_counts))
    representative_indices = tuple(sorted(_representative_owners(all_counts, band.end - band.start + 1)))
    if not representative_indices:
        representative_indices = all_indices

    representative_components = [components[index] for index in representative_indices]
    height, width = mask_shape
    content_left = min(component.left for component in representative_components)
    content_top = min(component.top for component in representative_components)
    content_right = max(component.right for component in representative_components)
    content_bottom = max(component.bottom for component in representative_components)
    band_start = band.start
    band_end = band.end

    if orientation == HORIZONTAL:
        left = 0
        right = width - 1
        top = band_start
        bottom = band_end
    else:
        left = band_start
        right = band_end
        top = 0
        bottom = height - 1

    representative_thickness = float((band_end - band_start) + 1)

    return LineCluster(
        component_indices=representative_indices,
        support_indices=all_indices,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        content_left=content_left,
        content_top=content_top,
        content_right=content_right,
        content_bottom=content_bottom,
        area=sum(component.area for component in representative_components),
        representative_thickness=representative_thickness,
    )


def _combine_scanline_bands(left: _ScanlineBand, right: _ScanlineBand) -> _ScanlineBand:
    return _ScanlineBand(
        start=left.start,
        end=right.end,
        slab_start=left.slab_start,
        slab_end=right.slab_end,
    )


def _select_furigana_components(
    lines: list[LineCluster],
    orientation: str,
    stats: OrientationLineStats | None,
) -> tuple[list[int], list[LineRole]]:
    if len(lines) < 2 or stats is None:
        return [], ["other" for _ in lines]

    main_line_indices = [
        index
        for index, line in enumerate(lines)
        if line.representative_thickness >= stats.main_min_thickness
    ]
    main_lines = [lines[index] for index in main_line_indices]
    if not main_lines:
        return [], ["other" for _ in lines]

    selected_line_indices: list[int] = []
    for index, line in enumerate(lines):
        if index in main_line_indices:
            continue

        thickness = line.representative_thickness
        if thickness < stats.furigana_min_thickness or thickness > stats.furigana_max_thickness:
            continue
        if _matches_main_line(candidate=line, main_lines=main_lines, orientation=orientation, stats=stats):
            selected_line_indices.append(index)

    line_roles: list[LineRole] = ["other" for _ in lines]
    for index in main_line_indices:
        line_roles[index] = "main"
    for index in selected_line_indices:
        line_roles[index] = "furigana"
    return selected_line_indices, line_roles


def _attach_furigana_fragments(
    lines: list[LineCluster],
    selected_line_indices: list[int],
    line_roles: list[LineRole],
    orientation: str,
    stats: OrientationLineStats | None,
) -> tuple[list[int], list[LineRole]]:
    if not selected_line_indices or stats is None:
        return selected_line_indices, line_roles

    selected = set(selected_line_indices)
    roles = list(line_roles)
    max_fragment_gap = max(2.0, stats.furigana_min_thickness * _FURIGANA_FRAGMENT_GAP_RATIO)

    for anchor_index in sorted(selected_line_indices):
        anchor = lines[anchor_index]
        for direction in (-1, 1):
            candidate_index = anchor_index + direction
            while 0 <= candidate_index < len(lines):
                if roles[candidate_index] == "main":
                    break
                if candidate_index in selected:
                    candidate_index += direction
                    continue

                candidate = lines[candidate_index]
                if not _is_furigana_fragment(
                    candidate=candidate,
                    anchor=anchor,
                    orientation=orientation,
                    max_fragment_gap=max_fragment_gap,
                ):
                    if _cross_gap(anchor, candidate, orientation) > max_fragment_gap:
                        break
                    candidate_index += direction
                    continue
                selected.add(candidate_index)
                roles[candidate_index] = "furigana"
                candidate_index += direction

    return sorted(selected), roles


def _is_furigana_fragment(
    candidate: LineCluster,
    anchor: LineCluster,
    orientation: str,
    max_fragment_gap: float,
) -> bool:
    candidate_thickness = candidate.representative_thickness
    anchor_thickness = anchor.representative_thickness
    if candidate_thickness >= anchor_thickness:
        return False
    if candidate_thickness > (anchor_thickness * 0.55):
        return False

    gap = _cross_gap(anchor, candidate, orientation)
    if gap > max_fragment_gap:
        return False

    overlap = _flow_overlap_ratio(candidate, anchor, orientation)
    flow_gap = _flow_gap(candidate, anchor, orientation)
    if overlap < _FURIGANA_FRAGMENT_MIN_OVERLAP and flow_gap > max(4.0, anchor_thickness * 0.25):
        return False

    return True


def _matches_main_line(
    candidate: LineCluster,
    main_lines: list[LineCluster],
    orientation: str,
    stats: OrientationLineStats,
) -> bool:
    for main_line in main_lines:
        if orientation == HORIZONTAL:
            gap = _line_gap(candidate.top, candidate.bottom, main_line.top, main_line.bottom)
        else:
            gap = _line_gap(candidate.left, candidate.right, main_line.left, main_line.right)
        if gap > stats.max_cross_gap:
            continue

        overlap_ratio = _flow_overlap_ratio(candidate, main_line, orientation)
        if overlap_ratio < stats.min_flow_overlap_ratio:
            continue
        return True

    return False


def _flow_overlap_ratio(candidate: LineCluster, main_line: LineCluster, orientation: str) -> float:
    start = max(candidate.flow_start(orientation), main_line.flow_start(orientation))
    end = min(candidate.flow_end(orientation), main_line.flow_end(orientation))
    if end < start:
        return 0.0

    overlap = end - start + 1
    candidate_length = candidate.flow_end(orientation) - candidate.flow_start(orientation) + 1
    main_length = main_line.flow_end(orientation) - main_line.flow_start(orientation) + 1
    denominator = max(1, min(candidate_length, main_length))
    return overlap / float(denominator)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = sorted_weights.sum() * percentile
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_percentile(values, weights, 0.50)


def _line_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if end_a < start_b:
        return start_b - end_a - 1
    if end_b < start_a:
        return start_a - end_b - 1
    return 0


def _flow_gap(left: LineCluster, right: LineCluster, orientation: str) -> int:
    return _line_gap(
        left.flow_start(orientation),
        left.flow_end(orientation),
        right.flow_start(orientation),
        right.flow_end(orientation),
    )


def _expand_line_boxes(
    lines: list[LineCluster],
    ink_mask: np.ndarray,
    orientation: str,
) -> list[tuple[int, int, int, int]]:
    if not lines:
        return []

    expanded: list[tuple[int, int, int, int]] = []
    if orientation == HORIZONTAL:
        height, width = ink_mask.shape
        for index, line in enumerate(lines):
            flow_left = max(0, min(width - 1, line.content_left))
            flow_right = max(flow_left, min(width - 1, line.content_right))
            row_active = np.any(ink_mask[:, flow_left : flow_right + 1], axis=1)
            top = max(0, line.top)
            bottom = min(height - 1, line.bottom)
            top_limit = _horizontal_top_limit(lines, index)
            bottom_limit = _horizontal_bottom_limit(lines, index, height)
            while top > top_limit and row_active[top - 1]:
                top -= 1
            while bottom < bottom_limit and row_active[bottom + 1]:
                bottom += 1
            expanded.append((0, top, width - 1, bottom))
        return expanded

    height, width = ink_mask.shape
    for index, line in enumerate(lines):
        flow_top = max(0, min(height - 1, line.content_top))
        flow_bottom = max(flow_top, min(height - 1, line.content_bottom))
        col_active = np.any(ink_mask[flow_top : flow_bottom + 1, :], axis=0)
        left = max(0, line.left)
        right = min(width - 1, line.right)
        left_limit = _vertical_left_limit(lines, index)
        right_limit = _vertical_right_limit(lines, index, width)
        while left > left_limit and col_active[left - 1]:
            left -= 1
        while right < right_limit and col_active[right + 1]:
            right += 1
        expanded.append((left, 0, right, height - 1))
    return expanded


def _horizontal_top_limit(lines: list[LineCluster], index: int) -> int:
    if index == 0:
        return 0
    previous = lines[index - 1]
    current = lines[index]
    return max(0, ((previous.bottom + current.top) // 2) + 1)


def _horizontal_bottom_limit(lines: list[LineCluster], index: int, height: int) -> int:
    if index + 1 >= len(lines):
        return height - 1
    current = lines[index]
    following = lines[index + 1]
    return min(height - 1, (current.bottom + following.top) // 2)


def _vertical_left_limit(lines: list[LineCluster], index: int) -> int:
    if index == 0:
        return 0
    previous = lines[index - 1]
    current = lines[index]
    return max(0, ((previous.right + current.left) // 2) + 1)


def _vertical_right_limit(lines: list[LineCluster], index: int, width: int) -> int:
    if index + 1 >= len(lines):
        return width - 1
    current = lines[index]
    following = lines[index + 1]
    return min(width - 1, (current.right + following.left) // 2)


def _build_selected_mask(
    selected_line_indices: list[int],
    boxes: list[tuple[int, int, int, int]],
    mask_shape: tuple[int, int],
) -> np.ndarray:
    selected_mask = np.zeros(mask_shape, dtype=bool)
    if not selected_line_indices:
        return selected_mask

    height, width = mask_shape
    for line_index in selected_line_indices:
        left, top, right, bottom = boxes[line_index]
        left = max(0, left)
        top = max(0, top)
        right = min(width - 1, right)
        bottom = min(height - 1, bottom)
        selected_mask[top : bottom + 1, left : right + 1] = True

    return selected_mask
