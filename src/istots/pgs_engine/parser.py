from __future__ import annotations

import atexit
from array import array
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Iterator

from PIL import Image

from .assembly import (
    DEDUPE_MAX_GAP_PTS,
    FrameCandidate as CandidateFrame,
    build_frame_candidates,
    dedupe_consecutive_identical,
    finalize_candidates,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - optional acceleration dependency
    np = None

PG_MAGIC = b"PG"

SEGMENT_PDS = 0x14
SEGMENT_ODS = 0x15
SEGMENT_PCS = 0x16
SEGMENT_WDS = 0x17
SEGMENT_END = 0x80

COMPOSITION_STATE_NORMAL = 0x00
COMPOSITION_STATE_ACQUISITION = 0x40
COMPOSITION_STATE_EPOCH_START = 0x80
COMPOSITION_STATE_EPOCH_CONTINUE = 0xC0
VALID_COMPOSITION_STATES = {
    COMPOSITION_STATE_NORMAL,
    COMPOSITION_STATE_ACQUISITION,
    COMPOSITION_STATE_EPOCH_START,
    COMPOSITION_STATE_EPOCH_CONTINUE,
}
START_STATES = {COMPOSITION_STATE_ACQUISITION, COMPOSITION_STATE_EPOCH_START}
CACHE_RESET_STATES = {COMPOSITION_STATE_EPOCH_START}

DISPLAY_CACHE_MAX_ITEMS = 4096
PREDECODE_MAX_AUTO_WORKERS = 16

_PREDECODE_POOL_LOCK = threading.Lock()
_PREDECODE_POOLS: dict[int, ProcessPoolExecutor] = {}


@dataclass
class PaletteEntry:
    y: int
    cr: int
    cb: int
    alpha: int


@dataclass
class CompositionObject:
    object_id: int
    window_id: int
    x: int
    y: int
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    cropped: bool = False


@dataclass
class PcsSegment:
    width: int
    height: int
    frame_rate: int
    composition_number: int
    composition_state: int
    palette_update_flag: int
    palette_id: int
    presentation_timestamp: int
    objects: list[CompositionObject]


@dataclass
class WindowInfo:
    window_id: int
    x: int
    y: int
    width: int
    height: int


@dataclass
class ObjectBuffer:
    object_id: int
    version: int = 0
    width: int = 0
    height: int = 0
    expected_data_len: int = 0
    data: bytearray = field(default_factory=bytearray)
    complete: bool = False
    decoded_indices: list[list[int]] | None = None
    data_digest: bytes | None = None


@dataclass
class ParsedDisplaySet:
    raw_index: int
    pcs: PcsSegment | None
    complete: bool
    decoded_pixels: list[list[int]] | None
    decoded_windows: tuple["DecodedWindow", ...] = ()


@dataclass(frozen=True)
class DecodedWindow:
    window_id: int
    left: int
    top: int
    right: int
    bottom: int
    object_ids: tuple[int, ...]
    pixels: list[list[int]]

    @property
    def size(self) -> tuple[int, int]:
        if not self.pixels:
            return (0, 0)
        return (len(self.pixels[0]), len(self.pixels))


@dataclass
class EngineFrame:
    raw_index: int
    start_pts: int
    end_pts: int
    start_ms: int
    end_ms: int
    image_hash: int
    pixels: list[list[int]]

    @property
    def size(self) -> tuple[int, int]:
        if not self.pixels:
            return (0, 0)
        return (len(self.pixels[0]), len(self.pixels))

    def to_image(self) -> Image.Image:
        width, height = self.size
        if width == 0 or height == 0:
            return Image.new("RGB", (1, 1), (255, 255, 255))

        flattened = bytearray(width * height)
        offset = 0
        for row in self.pixels:
            if len(row) != width:
                raise RuntimeError("decoded image has inconsistent row width")
            flattened[offset : offset + width] = bytes(row)
            offset += width
        return Image.frombytes("L", (width, height), bytes(flattened)).convert("RGB")


@dataclass
class _Packet:
    pts: int
    dts: int
    segment_type: int
    payload: bytes


@dataclass(frozen=True)
class _DecodedRaster:
    pixels: list[list[int]]
    left: int
    top: int
    right: int
    bottom: int


class PgsEngine:
    def __init__(self, input_sup: Path) -> None:
        self.input_sup = input_sup.expanduser().resolve()
        self._palettes: dict[int, dict[int, PaletteEntry]] = {}
        self._palette_versions: dict[int, int] = {}
        self._objects: dict[int, ObjectBuffer] = {}
        self._windows: dict[int, WindowInfo] = {}
        self._display_decode_cache: OrderedDict[tuple, tuple[DecodedWindow, ...]] = OrderedDict()
        self._predecoded_object_cache: dict[tuple[int, int, bytes], object] = {}

    def parse_display_sets(
        self,
        max_display_sets: int | None = None,
        predecode_workers: int = 0,
        include_decoded_pixels: bool = True,
    ) -> list[ParsedDisplaySet]:
        if not self.input_sup.exists():
            raise FileNotFoundError(f"Input SUP file not found: {self.input_sup}")
        if max_display_sets is not None and max_display_sets <= 0:
            raise ValueError("max_display_sets must be a positive integer")
        if predecode_workers < -1:
            raise ValueError("predecode_workers must be -1, 0, or a positive integer")

        self._palettes.clear()
        self._palette_versions.clear()
        self._objects.clear()
        self._windows.clear()
        self._display_decode_cache.clear()
        self._predecoded_object_cache.clear()

        sup_data: bytes | None = None
        if predecode_workers != 0:
            sup_data = self.input_sup.read_bytes()
            self._build_predecoded_object_cache(
                max_display_sets=max_display_sets,
                predecode_workers=predecode_workers,
                sup_data=sup_data,
            )

        rows: list[ParsedDisplaySet] = []
        for raw_index, packets in enumerate(_iter_display_set_packets(self.input_sup, sup_data=sup_data)):
            rows.append(
                self._parse_display_set(
                    raw_index,
                    packets,
                    include_decoded_pixels=include_decoded_pixels,
                )
            )
            if max_display_sets is not None and len(rows) >= max_display_sets:
                break
        return rows

    def build_frames(
        self,
        max_items: int | None = None,
        predecode_workers: int = -1,
    ) -> list[EngineFrame]:
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be a positive integer")

        display_sets = self.parse_display_sets(predecode_workers=predecode_workers)
        candidates = build_frame_candidates(display_sets, hash_pixels=_hash_pixels)
        finalized = finalize_candidates(candidates, max_items=max_items)
        deduped = dedupe_consecutive_identical(finalized, max_gap_pts=DEDUPE_MAX_GAP_PTS)

        frames: list[EngineFrame] = []
        for row in deduped:
            start_ms = _pts_to_ms(row.start_pts)
            end_ms = _pts_to_ms(row.end_pts)
            if end_ms <= start_ms:
                end_ms = start_ms + 1
            frames.append(
                EngineFrame(
                    raw_index=row.raw_index,
                    start_pts=row.start_pts,
                    end_pts=row.end_pts,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    image_hash=row.image_hash,
                    pixels=row.pixels,
                )
            )
        return frames

    def _parse_display_set(
        self,
        raw_index: int,
        packets: list[_Packet],
        *,
        include_decoded_pixels: bool = True,
    ) -> ParsedDisplaySet:
        pcs: PcsSegment | None = None
        saw_end = False

        for packet in packets:
            if packet.segment_type == SEGMENT_PCS:
                parsed_pcs = _parse_pcs(packet.payload, packet.pts)
                if parsed_pcs is None:
                    continue
                pcs = parsed_pcs
                if pcs.composition_state in CACHE_RESET_STATES:
                    # Start of a new epoch. Object/window state from the previous
                    # epoch should not bleed into the next one.
                    self._objects.clear()
                    self._windows.clear()

            elif packet.segment_type == SEGMENT_PDS:
                _update_palette(self._palettes, packet.payload, self._palette_versions)
            elif packet.segment_type == SEGMENT_ODS:
                _update_object(self._objects, packet.payload)
            elif packet.segment_type == SEGMENT_WDS:
                _update_windows(self._windows, packet.payload)
            elif packet.segment_type == SEGMENT_END:
                saw_end = True

        if pcs is None:
            return ParsedDisplaySet(
                raw_index=raw_index,
                pcs=None,
                complete=False,
                decoded_pixels=None,
            )

        if pcs.composition_state not in VALID_COMPOSITION_STATES:
            return ParsedDisplaySet(
                raw_index=raw_index,
                pcs=pcs,
                complete=False,
                decoded_pixels=None,
            )

        if not saw_end:
            return ParsedDisplaySet(
                raw_index=raw_index,
                pcs=pcs,
                complete=False,
                decoded_pixels=None,
            )

        if not pcs.objects:
            return ParsedDisplaySet(
                raw_index=raw_index,
                pcs=pcs,
                complete=True,
                decoded_pixels=None,
                decoded_windows=(),
            )

        decoded_windows = _decode_display_set_windows_cached(
            pcs=pcs,
            palettes=self._palettes,
            palette_versions=self._palette_versions,
            objects=self._objects,
            predecoded_objects=self._predecoded_object_cache,
            cache=self._display_decode_cache,
            max_cache_items=DISPLAY_CACHE_MAX_ITEMS,
        )
        decoded = _compose_decoded_windows(decoded_windows) if include_decoded_pixels else None
        return ParsedDisplaySet(
            raw_index=raw_index,
            pcs=pcs,
            complete=(decoded is not None) if include_decoded_pixels else bool(decoded_windows),
            decoded_pixels=decoded,
            decoded_windows=decoded_windows,
        )

    def _build_predecoded_object_cache(
        self,
        max_display_sets: int | None,
        predecode_workers: int,
        sup_data: bytes | None,
    ) -> None:
        worker_count = _resolve_predecode_workers(predecode_workers)
        if worker_count <= 0:
            return

        unique_objects = _collect_unique_completed_objects(
            path=self.input_sup,
            max_display_sets=max_display_sets,
            sup_data=sup_data,
        )
        if not unique_objects:
            return

        if worker_count > 1:
            # Avoid oversubscribing processes when the unique ODS set is small.
            max_useful_workers = max(1, len(unique_objects) // 16)
            worker_count = min(worker_count, max_useful_workers)

        if worker_count == 1:
            for key, (data, width, height) in unique_objects.items():
                self._predecoded_object_cache[key] = _decode_object_rle(data, width, height)
            return

        work_items = [
            (key, data, width, height)
            for key, (data, width, height) in unique_objects.items()
        ]
        chunk_size = max(1, len(work_items) // (worker_count * 8))

        executor = _get_predecode_pool(worker_count)
        for key, flat in executor.map(
            _predecode_worker_decode_flat,
            work_items,
            chunksize=chunk_size,
        ):
            width, height, _ = key
            self._predecoded_object_cache[key] = _flat_bytes_to_rows(flat, width, height)


def iter_engine_frames(
    input_sup: Path,
    max_items: int | None = None,
    predecode_workers: int = -1,
) -> Iterator[EngineFrame]:
    engine = PgsEngine(input_sup)
    yield from engine.build_frames(max_items=max_items, predecode_workers=predecode_workers)


def _iter_display_set_packets(path: Path, sup_data: bytes | None = None) -> Iterator[list[_Packet]]:
    data = sup_data if sup_data is not None else path.read_bytes()
    cursor = 0
    size = len(data)
    display_set: list[_Packet] = []

    while cursor + 13 <= size:
        if data[cursor : cursor + 2] != PG_MAGIC:
            raise RuntimeError(f"Invalid SUP packet signature at offset {cursor}")
        cursor += 2

        pts = int.from_bytes(data[cursor : cursor + 4], byteorder="big")
        cursor += 4
        dts = int.from_bytes(data[cursor : cursor + 4], byteorder="big")
        cursor += 4
        segment_type = data[cursor]
        cursor += 1
        payload_size = int.from_bytes(data[cursor : cursor + 2], byteorder="big")
        cursor += 2

        if cursor + payload_size > size:
            raise RuntimeError("SUP packet payload exceeds file size")
        payload = data[cursor : cursor + payload_size]
        cursor += payload_size

        display_set.append(
            _Packet(
                pts=pts,
                dts=dts,
                segment_type=segment_type,
                payload=payload,
            )
        )

        if segment_type == SEGMENT_END:
            yield display_set
            display_set = []

    if display_set:
        yield display_set


def _resolve_predecode_workers(predecode_workers: int) -> int:
    if predecode_workers == 0:
        return 0
    if predecode_workers == -1:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, PREDECODE_MAX_AUTO_WORKERS))
    if predecode_workers < -1:
        return 0
    return predecode_workers


def _get_predecode_pool(worker_count: int) -> ProcessPoolExecutor:
    with _PREDECODE_POOL_LOCK:
        pool = _PREDECODE_POOLS.get(worker_count)
        if pool is None:
            pool = ProcessPoolExecutor(max_workers=worker_count)
            _PREDECODE_POOLS[worker_count] = pool
        return pool


def shutdown_predecode_pools() -> None:
    with _PREDECODE_POOL_LOCK:
        pools = list(_PREDECODE_POOLS.values())
        _PREDECODE_POOLS.clear()
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=False)

atexit.register(shutdown_predecode_pools)


def _collect_unique_completed_objects(
    path: Path,
    max_display_sets: int | None,
    sup_data: bytes | None = None,
) -> dict[tuple[int, int, bytes], tuple[bytes, int, int]]:
    objects: dict[int, ObjectBuffer] = {}
    unique: dict[tuple[int, int, bytes], tuple[bytes, int, int]] = {}

    for display_set_index, packets in enumerate(_iter_display_set_packets(path, sup_data=sup_data)):
        for packet in packets:
            if packet.segment_type == SEGMENT_PCS:
                if _is_reset_pcs_payload(packet.payload):
                    objects.clear()
                continue

            if packet.segment_type != SEGMENT_ODS:
                continue
            _update_object(objects, packet.payload)

            if len(packet.payload) < 2:
                continue
            object_id = int.from_bytes(packet.payload[0:2], byteorder="big")
            obj = objects.get(object_id)
            if obj is None or not obj.complete:
                continue
            if obj.width <= 0 or obj.height <= 0 or not obj.data:
                continue
            digest = _get_object_data_digest(obj)
            key = (obj.width, obj.height, digest)
            if key in unique:
                continue
            unique[key] = (bytes(obj.data), obj.width, obj.height)

        if max_display_sets is not None and display_set_index + 1 >= max_display_sets:
            break

    return unique


def _is_reset_pcs_payload(payload: bytes) -> bool:
    # PCS layout: composition_state is byte index 7.
    if len(payload) < 8:
        return False
    return payload[7] in CACHE_RESET_STATES


def _predecode_worker_decode_flat(
    item: tuple[tuple[int, int, bytes], bytes, int, int],
) -> tuple[tuple[int, int, bytes], bytes]:
    key, data, width, height = item
    if np is None:
        rows = _decode_object_rle_python(data, width, height)
        flat = b"".join(bytes(row) for row in rows)
    else:
        flat = _decode_object_rle_flat_np_repeat(data, width, height)
    return key, flat


def _flat_bytes_to_rows(flat: bytes, width: int, height: int) -> list[list[int]]:
    if width <= 0 or height <= 0:
        return []
    total_pixels = width * height
    if len(flat) > total_pixels:
        flat = flat[:total_pixels]
    elif len(flat) < total_pixels:
        flat = flat + (b"\x00" * (total_pixels - len(flat)))
    if np is not None:
        array_2d = np.frombuffer(flat, dtype=np.uint8)
        return array_2d.reshape((height, width))
    flat_view = memoryview(flat)
    rows: list[list[int]] = []
    for row in range(height):
        start = row * width
        end = start + width
        rows.append(flat_view[start:end])
    return rows


def _parse_pcs(payload: bytes, pts: int) -> PcsSegment | None:
    if len(payload) < 11:
        return None

    width = int.from_bytes(payload[0:2], byteorder="big")
    height = int.from_bytes(payload[2:4], byteorder="big")
    frame_rate = payload[4]
    composition_number = int.from_bytes(payload[5:7], byteorder="big")
    composition_state = payload[7]
    palette_update_flag = payload[8]
    palette_id = payload[9]
    object_count = payload[10]

    objects: list[CompositionObject] = []
    cursor = 11
    for _ in range(object_count):
        if cursor + 8 > len(payload):
            break

        object_id = int.from_bytes(payload[cursor : cursor + 2], byteorder="big")
        window_id = payload[cursor + 2]
        object_flags = payload[cursor + 3]
        x = int.from_bytes(payload[cursor + 4 : cursor + 6], byteorder="big")
        y = int.from_bytes(payload[cursor + 6 : cursor + 8], byteorder="big")
        cursor += 8

        cropped = bool(object_flags & 0x40 or object_flags & 0x80)
        crop_x = 0
        crop_y = 0
        crop_width = 0
        crop_height = 0
        if cropped and cursor + 8 <= len(payload):
            crop_x = int.from_bytes(payload[cursor : cursor + 2], byteorder="big")
            crop_y = int.from_bytes(payload[cursor + 2 : cursor + 4], byteorder="big")
            crop_width = int.from_bytes(payload[cursor + 4 : cursor + 6], byteorder="big")
            crop_height = int.from_bytes(payload[cursor + 6 : cursor + 8], byteorder="big")
            cursor += 8

        objects.append(
            CompositionObject(
                object_id=object_id,
                window_id=window_id,
                x=x,
                y=y,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_width=crop_width,
                crop_height=crop_height,
                cropped=cropped,
            )
        )

    return PcsSegment(
        width=width,
        height=height,
        frame_rate=frame_rate,
        composition_number=composition_number,
        composition_state=composition_state,
        palette_update_flag=palette_update_flag,
        palette_id=palette_id,
        presentation_timestamp=pts,
        objects=objects,
    )


def _update_palette(
    palettes: dict[int, dict[int, PaletteEntry]],
    payload: bytes,
    palette_versions: dict[int, int] | None = None,
) -> None:
    if len(payload) < 2:
        return

    palette_id = payload[0]
    entries = palettes.setdefault(palette_id, {})
    changed = False
    cursor = 2

    while cursor + 5 <= len(payload):
        index = payload[cursor]
        y = payload[cursor + 1]
        cr = payload[cursor + 2]
        cb = payload[cursor + 3]
        alpha = payload[cursor + 4]
        current = entries.get(index)
        if (
            current is not None
            and current.y == y
            and current.cr == cr
            and current.cb == cb
            and current.alpha == alpha
        ):
            cursor += 5
            continue
        entries[index] = PaletteEntry(y=y, cr=cr, cb=cb, alpha=alpha)
        changed = True
        cursor += 5

    if changed and palette_versions is not None:
        palette_versions[palette_id] = palette_versions.get(palette_id, 0) + 1

def _update_object(objects: dict[int, ObjectBuffer], payload: bytes) -> None:
    if len(payload) < 7:
        return

    object_id = int.from_bytes(payload[0:2], byteorder="big")
    version = payload[2]
    sequence_flag = payload[3]
    object_data_len = int.from_bytes(payload[4:7], byteorder="big")
    first_in_sequence = bool(sequence_flag & 0x80)
    last_in_sequence = bool(sequence_flag & 0x40)

    if first_in_sequence:
        if len(payload) < 11:
            return

        width = int.from_bytes(payload[7:9], byteorder="big")
        height = int.from_bytes(payload[9:11], byteorder="big")
        fragment = payload[11:]
        expected_data_len = max(0, object_data_len - 4)

        if last_in_sequence and len(fragment) >= expected_data_len:
            existing = objects.get(object_id)
            if (
                existing is not None
                and existing.complete
                and existing.version == version
                and existing.width == width
                and existing.height == height
                and existing.expected_data_len == expected_data_len
                and len(existing.data) == len(fragment)
                and existing.data == fragment
            ):
                return

        obj = ObjectBuffer(
            object_id=object_id,
            version=version,
            width=width,
            height=height,
            expected_data_len=expected_data_len,
            data=bytearray(fragment),
            complete=False,
            decoded_indices=None,
            data_digest=None,
        )
        if last_in_sequence and len(obj.data) >= obj.expected_data_len:
            obj.complete = True
        objects[object_id] = obj
        return

    obj = objects.get(object_id)
    if obj is None:
        return

    obj.version = version
    fragment = payload[7:]
    obj.data.extend(fragment)
    obj.decoded_indices = None
    obj.data_digest = None
    if last_in_sequence and len(obj.data) >= obj.expected_data_len:
        obj.complete = True


def _update_windows(windows: dict[int, WindowInfo], payload: bytes) -> None:
    if len(payload) < 1:
        return

    count = payload[0]
    cursor = 1
    for _ in range(count):
        if cursor + 9 > len(payload):
            break

        window_id = payload[cursor]
        x = int.from_bytes(payload[cursor + 1 : cursor + 3], byteorder="big")
        y = int.from_bytes(payload[cursor + 3 : cursor + 5], byteorder="big")
        width = int.from_bytes(payload[cursor + 5 : cursor + 7], byteorder="big")
        height = int.from_bytes(payload[cursor + 7 : cursor + 9], byteorder="big")
        windows[window_id] = WindowInfo(window_id=window_id, x=x, y=y, width=width, height=height)
        cursor += 9


def _decode_display_set(
    pcs: PcsSegment,
    palettes: dict[int, dict[int, PaletteEntry]],
    palette_versions: dict[int, int],
    objects: dict[int, ObjectBuffer],
    predecoded_objects: dict[tuple[int, int, bytes], object] | None = None,
) -> list[list[int]] | None:
    decoded_windows = _decode_display_set_windows(
        pcs=pcs,
        palettes=palettes,
        palette_versions=palette_versions,
        objects=objects,
        predecoded_objects=predecoded_objects,
    )
    return _compose_decoded_windows(decoded_windows)


def _compose_decoded_windows(decoded_windows: tuple[DecodedWindow, ...]) -> list[list[int]] | None:
    if not decoded_windows:
        return None

    min_left = min(window.left for window in decoded_windows)
    min_top = min(window.top for window in decoded_windows)
    max_right = max(window.left + window.size[0] for window in decoded_windows)
    max_bottom = max(window.top + window.size[1] for window in decoded_windows)
    if max_right <= min_left or max_bottom <= min_top:
        return None

    out_width = max_right - min_left
    out_height = max_bottom - min_top
    white_row = b"\xFF" * out_width
    output: list[bytearray] = [bytearray(white_row) for _ in range(out_height)]

    for window in decoded_windows:
        width, height = window.size
        if width <= 0 or height <= 0:
            continue
        dst_x_offset = window.left - min_left
        dst_y_offset = window.top - min_top
        for row_index in range(height):
            src_row = window.pixels[row_index]
            out_row = output[dst_y_offset + row_index]
            for col_index in range(width):
                value = src_row[col_index]
                if value == 255:
                    continue
                out_row[dst_x_offset + col_index] = value

    if all(all(value == 255 for value in row) for row in output):
        return None

    return [list(row) for row in output]


def _decode_display_set_windows(
    pcs: PcsSegment,
    palettes: dict[int, dict[int, PaletteEntry]],
    palette_versions: dict[int, int],
    objects: dict[int, ObjectBuffer],
    predecoded_objects: dict[tuple[int, int, bytes], object] | None = None,
) -> tuple[DecodedWindow, ...]:
    if not pcs.objects:
        return ()

    palette = palettes.get(pcs.palette_id)
    if not palette:
        return ()
    gray_lut, alpha_lut = _build_palette_lut(palette)

    refs_by_window: dict[int, list[CompositionObject]] = {}
    for ref in pcs.objects:
        refs_by_window.setdefault(ref.window_id, []).append(ref)

    decoded_windows: list[DecodedWindow] = []
    for window_id in sorted(refs_by_window):
        decoded = _decode_object_refs(
            object_refs=refs_by_window[window_id],
            gray_lut=gray_lut,
            alpha_lut=alpha_lut,
            objects=objects,
            predecoded_objects=predecoded_objects,
        )
        if decoded is None:
            continue
        decoded_windows.append(
            DecodedWindow(
                window_id=window_id,
                left=decoded.left,
                top=decoded.top,
                right=decoded.right,
                bottom=decoded.bottom,
                object_ids=tuple(ref.object_id for ref in refs_by_window[window_id]),
                pixels=decoded.pixels,
            )
        )
    return tuple(decoded_windows)


def _decode_object_refs(
    object_refs: list[CompositionObject],
    gray_lut: list[int],
    alpha_lut: list[int],
    objects: dict[int, ObjectBuffer],
    predecoded_objects: dict[tuple[int, int, bytes], object] | None = None,
) -> _DecodedRaster | None:
    prepared: list[tuple[int, int, list[list[int]], int, int, int, int]] = []
    for ref in object_refs:
        obj = objects.get(ref.object_id)
        if obj is None or not obj.complete:
            return None
        if obj.width <= 0 or obj.height <= 0:
            return None

        if obj.decoded_indices is None:
            decoded_from_cache: object | None = None
            if predecoded_objects is not None:
                digest = _get_object_data_digest(obj)
                decoded_from_cache = predecoded_objects.get((obj.width, obj.height, digest))
            if decoded_from_cache is not None:
                obj.decoded_indices = decoded_from_cache
            else:
                obj.decoded_indices = _decode_object_rle(obj.data, obj.width, obj.height)
        indices = obj.decoded_indices

        src_x = 0
        src_y = 0
        src_w = obj.width
        src_h = obj.height
        if ref.cropped:
            src_x = max(0, ref.crop_x)
            src_y = max(0, ref.crop_y)
            src_w = ref.crop_width if ref.crop_width > 0 else (obj.width - src_x)
            src_h = ref.crop_height if ref.crop_height > 0 else (obj.height - src_y)
            src_w = min(src_w, obj.width - src_x)
            src_h = min(src_h, obj.height - src_y)

        if src_w <= 0 or src_h <= 0:
            continue

        prepared.append((ref.x, ref.y, indices, src_x, src_y, src_w, src_h))

    if not prepared:
        return None

    min_x = min(item[0] for item in prepared)
    min_y = min(item[1] for item in prepared)
    max_x = max(item[0] + item[5] for item in prepared)
    max_y = max(item[1] + item[6] for item in prepared)
    if max_x <= min_x or max_y <= min_y:
        return None

    out_width = max_x - min_x
    out_height = max_y - min_y
    if np is not None:
        decoded = _decode_display_set_numpy(
            prepared=prepared,
            out_width=out_width,
            out_height=out_height,
            min_x=min_x,
            min_y=min_y,
            gray_lut=gray_lut,
            alpha_lut=alpha_lut,
        )
        if decoded is not None:
            return decoded

    white_row = b"\xFF" * out_width
    output: list[bytearray] = [bytearray(white_row) for _ in range(out_height)]

    for dst_x, dst_y, indices, src_x, src_y, src_w, src_h in prepared:
        dst_x_offset = dst_x - min_x
        dst_y_offset = dst_y - min_y
        for row in range(src_h):
            src_row = indices[src_y + row]
            out_row = output[dst_y_offset + row]
            out_col = dst_x_offset
            src_col = src_x
            src_end = src_x + src_w
            while src_col < src_end:
                idx = src_row[src_col]
                alpha = alpha_lut[idx]
                if alpha > 0:
                    # pgs-parse grayscale output maps subtitle luminance as inverted Y.
                    gray = gray_lut[idx]
                    current = out_row[out_col]
                    out_row[out_col] = ((alpha * gray) + ((255 - alpha) * current) + 254) // 255
                src_col += 1
                out_col += 1

    # Match pgs-parse behavior: trim canvas to effective non-background area.
    trim_left = out_width
    trim_top = out_height
    trim_right = -1
    trim_bottom = -1
    for y, row in enumerate(output):
        for x, value in enumerate(row):
            if value == 255:
                continue
            if x < trim_left:
                trim_left = x
            if y < trim_top:
                trim_top = y
            if x > trim_right:
                trim_right = x
            if y > trim_bottom:
                trim_bottom = y

    if trim_right < trim_left or trim_bottom < trim_top:
        return None

    cropped: list[list[int]] = []
    for y in range(trim_top, trim_bottom + 1):
        row = output[y][trim_left : trim_right + 1]
        cropped.append(row)
    return _DecodedRaster(
        pixels=cropped,
        left=min_x + trim_left,
        top=min_y + trim_top,
        right=min_x + trim_right,
        bottom=min_y + trim_bottom,
    )


def _decode_display_set_windows_cached(
    pcs: PcsSegment,
    palettes: dict[int, dict[int, PaletteEntry]],
    palette_versions: dict[int, int],
    objects: dict[int, ObjectBuffer],
    predecoded_objects: dict[tuple[int, int, bytes], object],
    cache: OrderedDict[tuple, tuple[DecodedWindow, ...]],
    max_cache_items: int,
) -> tuple[DecodedWindow, ...]:
    key = _make_display_signature_key(
        pcs=pcs,
        palettes=palettes,
        palette_versions=palette_versions,
        objects=objects,
    )
    if key is not None:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    decoded = _decode_display_set_windows(
        pcs=pcs,
        palettes=palettes,
        palette_versions=palette_versions,
        objects=objects,
        predecoded_objects=predecoded_objects,
    )
    if key is not None and decoded:
        cache[key] = decoded
        cache.move_to_end(key)
        while len(cache) > max_cache_items:
            cache.popitem(last=False)
    return decoded


def _make_display_signature_key(
    pcs: PcsSegment,
    palettes: dict[int, dict[int, PaletteEntry]],
    palette_versions: dict[int, int],
    objects: dict[int, ObjectBuffer],
) -> tuple | None:
    if not pcs.objects:
        return None

    palette = palettes.get(pcs.palette_id)
    if not palette:
        return None

    palette_version = int(palette_versions.get(pcs.palette_id, 0))

    object_part: list[tuple] = []
    for ref in pcs.objects:
        obj = objects.get(ref.object_id)
        if obj is None or not obj.complete or obj.width <= 0 or obj.height <= 0:
            return None
        data_digest = _get_object_data_digest(obj)
        object_part.append(
            (
                int(ref.object_id),
                int(ref.window_id),
                int(ref.x),
                int(ref.y),
                int(ref.crop_x),
                int(ref.crop_y),
                int(ref.crop_width),
                int(ref.crop_height),
                int(ref.cropped),
                int(obj.width),
                int(obj.height),
                data_digest,
            )
        )

    return (
        int(pcs.palette_id),
        palette_version,
        tuple(object_part),
    )


def _get_object_data_digest(obj: ObjectBuffer) -> bytes:
    cached = obj.data_digest
    if cached is not None:
        return cached
    digest = hashlib.blake2b(obj.data, digest_size=8).digest()
    obj.data_digest = digest
    return digest


def _decode_object_rle(data: bytes | bytearray, width: int, height: int) -> list[list[int]]:
    if width <= 0 or height <= 0:
        return []
    if np is None:
        return _decode_object_rle_python(data, width, height)

    flat = _decode_object_rle_flat_np_repeat(data, width, height)
    return _flat_bytes_to_rows(flat, width, height)


def _decode_object_rle_flat_np_repeat(data: bytes | bytearray, width: int, height: int) -> bytes:
    total_pixels = width * height
    run_lengths = array("I")
    run_colors = bytearray()
    append_len = run_lengths.append
    append_color = run_colors.append
    produced = 0

    cursor = 0
    x = 0
    y = 0
    data_len = len(data)

    while cursor < data_len and y < height:
        value = data[cursor]
        cursor += 1

        run_len = 1
        color = value
        if value == 0:
            if cursor >= data_len:
                break
            control = data[cursor]
            cursor += 1

            if control == 0:
                # End-of-line: explicit fill to line end for deterministic shape.
                if x < width:
                    remaining = width - x
                    if remaining > 0:
                        append_len(remaining)
                        append_color(0)
                        produced += remaining
                x = 0
                y += 1
                continue

            if control & 0x40:
                if cursor >= data_len:
                    break
                run_len = ((control & 0x3F) << 8) | data[cursor]
                cursor += 1
            else:
                run_len = control & 0x3F

            if control & 0x80:
                if cursor >= data_len:
                    break
                color = data[cursor]
                cursor += 1
            else:
                color = 0

        if run_len <= 0:
            continue

        while run_len > 0 and y < height:
            if x >= width:
                x = 0
                y += 1
                continue
            span = width - x
            if run_len < span:
                span = run_len
            append_len(span)
            append_color(color)
            produced += span
            x += span
            run_len -= span

    if produced < total_pixels:
        trailing = total_pixels - produced
        append_len(trailing)
        append_color(0)
        produced = total_pixels

    if len(run_lengths) == 0:
        return b"\x00" * total_pixels

    lengths_arr = np.frombuffer(run_lengths, dtype=np.uint32)
    colors_arr = np.frombuffer(run_colors, dtype=np.uint8)
    flat = np.repeat(colors_arr, lengths_arr)

    if flat.size > produced:
        flat = flat[:produced]
    if flat.size < produced:
        pad = np.zeros(produced - flat.size, dtype=np.uint8)
        flat = np.concatenate((flat, pad), axis=0)
    return flat.tobytes()


def _decode_object_rle_python(data: bytes | bytearray, width: int, height: int) -> list[list[int]]:
    rows: list[bytearray] = [bytearray(width) for _ in range(height)]

    cursor = 0
    x = 0
    y = 0
    data_len = len(data)

    while cursor < data_len and y < height:
        value = data[cursor]
        cursor += 1

        run_len = 1
        color = value
        if value == 0:
            if cursor >= data_len:
                break
            control = data[cursor]
            cursor += 1

            if control == 0:
                x = 0
                y += 1
                continue

            if control & 0x40:
                if cursor >= data_len:
                    break
                run_len = ((control & 0x3F) << 8) | data[cursor]
                cursor += 1
            else:
                run_len = control & 0x3F

            if control & 0x80:
                if cursor >= data_len:
                    break
                color = data[cursor]
                cursor += 1
            else:
                color = 0

        if run_len <= 0:
            continue
        fill_byte = bytes((color,)) if color else b""
        while run_len > 0 and y < height:
            if x >= width:
                x = 0
                y += 1
                continue
            span = width - x
            if run_len < span:
                span = run_len
            if color:
                rows[y][x : x + span] = fill_byte * span
            x += span
            run_len -= span

    return rows


def _decode_display_set_numpy(
    prepared: list[tuple[int, int, list[list[int]], int, int, int, int]],
    out_width: int,
    out_height: int,
    min_x: int,
    min_y: int,
    gray_lut: list[int],
    alpha_lut: list[int],
) -> _DecodedRaster | None:
    if np is None:
        return None

    output = np.full((out_height, out_width), 255, dtype=np.uint16)
    gray_arr = np.asarray(gray_lut, dtype=np.uint16)
    alpha_arr = np.asarray(alpha_lut, dtype=np.uint16)

    for dst_x, dst_y, indices, src_x, src_y, src_w, src_h in prepared:
        dst_x_offset = dst_x - min_x
        dst_y_offset = dst_y - min_y
        src_end = src_x + src_w

        if isinstance(indices, np.ndarray):
            idx_block = indices[src_y : src_y + src_h, src_x:src_end]
            alpha_block = alpha_arr[idx_block]
            out_block = output[
                dst_y_offset : dst_y_offset + src_h,
                dst_x_offset : dst_x_offset + src_w,
            ]
            gray_block = gray_arr[idx_block]
            blended_block = (
                (alpha_block * gray_block) + ((255 - alpha_block) * out_block) + 254
            ) // 255
            np.copyto(out_block, blended_block, where=(alpha_block > 0))
            continue

        for row in range(src_h):
            src_row = indices[src_y + row]
            if isinstance(src_row, (bytes, bytearray, memoryview)):
                idx_row = np.frombuffer(src_row, dtype=np.uint8)[src_x:src_end]
            else:
                idx_row = np.asarray(src_row[src_x:src_end], dtype=np.uint8)

            alpha_row = alpha_arr[idx_row]
            out_row = output[dst_y_offset + row, dst_x_offset : dst_x_offset + src_w]
            gray_row = gray_arr[idx_row]
            blended = ((alpha_row * gray_row) + ((255 - alpha_row) * out_row) + 254) // 255
            np.copyto(out_row, blended, where=(alpha_row > 0))

    non_white = output != 255
    rows_non_white = np.any(non_white, axis=1)
    if not np.any(rows_non_white):
        return None

    cols_non_white = np.any(non_white, axis=0)
    trim_top = int(rows_non_white.argmax())
    trim_bottom = int(len(rows_non_white) - 1 - rows_non_white[::-1].argmax())
    trim_left = int(cols_non_white.argmax())
    trim_right = int(len(cols_non_white) - 1 - cols_non_white[::-1].argmax())

    cropped = output[trim_top : trim_bottom + 1, trim_left : trim_right + 1].astype(np.uint8)
    return _DecodedRaster(
        pixels=[bytearray(row.tobytes()) for row in cropped],
        left=min_x + trim_left,
        top=min_y + trim_top,
        right=min_x + trim_right,
        bottom=min_y + trim_bottom,
    )


def _build_palette_lut(palette: dict[int, PaletteEntry]) -> tuple[list[int], list[int]]:
    gray_lut = [255] * 256
    alpha_lut = [0] * 256
    for idx, entry in palette.items():
        key = idx & 0xFF
        # pgs-parse grayscale output maps subtitle luminance as inverted Y.
        gray_lut[key] = 255 - int(entry.y)
        alpha_lut[key] = int(entry.alpha)
    # Empirically aligned with pgs-parse grayscale decode behavior:
    # palette index 30 acts as non-rendered halo in tested SUP streams.
    alpha_lut[0] = 0
    alpha_lut[30] = 0
    return gray_lut, alpha_lut


def _pts_to_ms(pts: int) -> int:
    return (pts + 45) // 90


def _hash_pixels(pixels: list[list[int]]) -> int:
    digest = hashlib.blake2b(digest_size=8)
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0
    digest.update(height.to_bytes(4, byteorder="big", signed=False))
    digest.update(width.to_bytes(4, byteorder="big", signed=False))
    for row in pixels:
        digest.update(bytes(row))
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def hash_gray_pixels(pixels: list[list[int]]) -> int:
    return _hash_pixels(pixels)
