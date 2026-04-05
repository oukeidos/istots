from __future__ import annotations

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
    analyses = [_analyze_image(image) for image in images]
    global_stats = _build_global_line_stats(analyses)
    return [_finalize_analysis(analysis, global_stats) for analysis in analyses]


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
    lines = _build_lines(components, labels, core_mask.shape, orientation)
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
    values = np.asarray(foreground_values, dtype=np.float64)
    if values.size == 0:
        return _THRESHOLD

    low = float(values.min())
    high = float(values.max())
    if high - low < 1.0:
        return int(min(_VISIBLE_THRESHOLD - 1, round(high)))

    centers = np.asarray(
        [
            np.percentile(values, 15.0),
            np.percentile(values, 85.0),
        ],
        dtype=np.float64,
    )
    if abs(float(centers[1] - centers[0])) < 1.0:
        centers = np.asarray([low, high], dtype=np.float64)

    labels = np.zeros(values.shape[0], dtype=np.int8)
    for _ in range(_CORE_CLUSTER_STEPS):
        distances = np.abs(values[:, np.newaxis] - centers[np.newaxis, :])
        labels = np.argmin(distances, axis=1).astype(np.int8)
        updated = centers.copy()
        for idx in (0, 1):
            cluster_values = values[labels == idx]
            if cluster_values.size > 0:
                updated[idx] = cluster_values.mean()
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated

    dark_center, light_center = sorted(float(value) for value in centers.tolist())
    if (light_center - dark_center) < _CORE_CLUSTER_GAP:
        return int(min(_VISIBLE_THRESHOLD - 1, round(light_center)))
    return int(min(_VISIBLE_THRESHOLD - 1, np.floor((dark_center + light_center) / 2.0)))


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
    components: list[Component] = []
    next_label = 1

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue

            stack = [(x, y)]
            labels[y, x] = next_label
            left = right = x
            top = bottom = y
            area = 0

            while stack:
                current_x, current_y = stack.pop()
                area += 1
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)

                for neighbor_x, neighbor_y in _neighbors(current_x, current_y, width, height):
                    if not mask[neighbor_y, neighbor_x]:
                        continue
                    if labels[neighbor_y, neighbor_x] != 0:
                        continue
                    labels[neighbor_y, neighbor_x] = next_label
                    stack.append((neighbor_x, neighbor_y))

            if area >= _MIN_COMPONENT_AREA:
                components.append(
                    Component(
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                        area=area,
                    )
                )
                next_label += 1
                continue

            labels[labels == next_label] = 0

    return components, labels


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
    labels: np.ndarray,
    mask_shape: tuple[int, int],
    orientation: str,
) -> list[LineCluster]:
    if not components:
        return []

    thicknesses = np.asarray([component.thickness(orientation) for component in components], dtype=np.float64)
    weights = np.asarray([component.area for component in components], dtype=np.float64)
    median_thickness = _weighted_median(thicknesses, weights)
    owner_sets = _build_scanline_owner_sets(labels, orientation)
    bands = _build_scanline_bands(owner_sets)
    if not bands:
        return []

    bands = _merge_scanline_bands(
        bands=bands,
        owner_sets=owner_sets,
        components=components,
        orientation=orientation,
        median_thickness=median_thickness,
    )
    return [
        _make_line_cluster_from_band(
            band=band,
            owner_sets=owner_sets,
            components=components,
            mask_shape=mask_shape,
            orientation=orientation,
        )
        for band in bands
    ]


def _build_scanline_owner_sets(labels: np.ndarray, orientation: str) -> list[frozenset[int]]:
    axis_length = labels.shape[0] if orientation == HORIZONTAL else labels.shape[1]
    owner_sets: list[frozenset[int]] = []
    for position in range(axis_length):
        slice_labels = labels[position, :] if orientation == HORIZONTAL else labels[:, position]
        unique_labels = np.unique(slice_labels)
        owner_sets.append(
            frozenset(int(label - 1) for label in unique_labels.tolist() if label != 0)
        )
    return owner_sets


def _build_scanline_bands(owner_sets: list[frozenset[int]]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    previous = frozenset()

    for position, owners in enumerate(owner_sets):
        if not owners:
            if start is not None:
                bands.append((start, position - 1))
                start = None
            previous = frozenset()
            continue

        if start is None:
            start = position
            previous = owners
            continue

        if _owner_set_overlap(previous, owners) >= _SCANLINE_MATCH_THRESHOLD:
            previous = owners
            continue

        bands.append((start, position - 1))
        start = position
        previous = owners

    if start is not None:
        bands.append((start, len(owner_sets) - 1))
    return bands


def _merge_scanline_bands(
    bands: list[tuple[int, int]],
    owner_sets: list[frozenset[int]],
    components: list[Component],
    orientation: str,
    median_thickness: float,
) -> list[tuple[int, int]]:
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
                gap = following[0] - current[1] - 1
                overlap = _band_overlap_signature(current, following, owner_sets)
                if gap <= merge_gap and (
                    overlap >= _SIGNATURE_OVERLAP_THRESHOLD
                    or (
                        gap == 0
                        and _band_scale_similarity(current, following, owner_sets, components, orientation)
                        >= _THICKNESS_SIMILARITY_THRESHOLD
                    )
                ):
                    next_bands.append((current[0], following[1]))
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
            thickness = current[1] - current[0] + 1
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
                    or (previous[1] + 1 == current[0] and previous_similarity >= _THICKNESS_SIMILARITY_THRESHOLD)
                ):
                    next_bands[-1] = (previous[0], current[1])
                    index += 1
                    changed = True
                    continue

                if following and (
                    following_overlap >= _SIGNATURE_OVERLAP_THRESHOLD
                    or (current[1] + 1 == following[0] and following_similarity >= _THICKNESS_SIMILARITY_THRESHOLD)
                ):
                    next_bands.append((current[0], following[1]))
                    index += 2
                    changed = True
                    continue
            next_bands.append(current)
            index += 1
        merged = next_bands

    return merged


def _band_overlap_signature(
    left_band: tuple[int, int] | None,
    right_band: tuple[int, int] | None,
    owner_sets: list[frozenset[int]],
) -> float:
    if left_band is None or right_band is None:
        return 0.0

    left_signature = _representative_owners(_band_owner_counts(left_band, owner_sets), left_band[1] - left_band[0] + 1)
    right_signature = _representative_owners(_band_owner_counts(right_band, owner_sets), right_band[1] - right_band[0] + 1)
    return _owner_set_overlap(left_signature, right_signature)


def _band_scale_similarity(
    left_band: tuple[int, int] | None,
    right_band: tuple[int, int] | None,
    owner_sets: list[frozenset[int]],
    components: list[Component],
    orientation: str,
) -> float:
    if left_band is None or right_band is None:
        return 0.0

    del owner_sets, components, orientation
    left_scale = float((left_band[1] - left_band[0]) + 1)
    right_scale = float((right_band[1] - right_band[0]) + 1)
    if left_scale <= 0 or right_scale <= 0:
        return 0.0
    return min(left_scale, right_scale) / max(left_scale, right_scale)


def _band_owner_counts(
    band: tuple[int, int],
    owner_sets: list[frozenset[int]],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    start, end = band
    for position in range(start, end + 1):
        for index in owner_sets[position]:
            counts[index] = counts.get(index, 0) + 1
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
    band: tuple[int, int],
    owner_sets: list[frozenset[int]],
    components: list[Component],
    mask_shape: tuple[int, int],
    orientation: str,
) -> LineCluster:
    all_counts = _band_owner_counts(band, owner_sets)
    all_indices = tuple(sorted(all_counts))
    representative_indices = tuple(sorted(_representative_owners(all_counts, band[1] - band[0] + 1)))
    if not representative_indices:
        representative_indices = all_indices

    line_components = [components[index] for index in all_indices]
    representative_components = [components[index] for index in representative_indices]
    height, width = mask_shape
    content_left = min(component.left for component in representative_components)
    content_top = min(component.top for component in representative_components)
    content_right = max(component.right for component in representative_components)
    content_bottom = max(component.bottom for component in representative_components)
    band_start, band_end = band

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
