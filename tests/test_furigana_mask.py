from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from istots.furigana_mask import HORIZONTAL, VERTICAL, build_furigana_mask, build_furigana_masks


def _make_horizontal_sample() -> Image.Image:
    image = Image.new("RGB", (140, 100), "white")
    draw = ImageDraw.Draw(image)

    for left in (18, 46, 74):
        draw.rectangle((left, 46, left + 17, 82), fill="black")
    for left in (21, 49, 77):
        draw.rectangle((left, 24, left + 8, 36), fill="black")
    return image


def _make_vertical_sample() -> Image.Image:
    image = Image.new("RGB", (120, 150), "white")
    draw = ImageDraw.Draw(image)

    for top in (18, 54, 90):
        draw.rectangle((34, top, 68, top + 17), fill="black")
    for top in (21, 57, 93):
        draw.rectangle((78, top, 88, top + 8), fill="black")
    return image


def _make_horizontal_bottom_sample() -> Image.Image:
    image = Image.new("RGB", (140, 110), "white")
    draw = ImageDraw.Draw(image)

    for left in (18, 46, 74):
        draw.rectangle((left, 22, left + 17, 58), fill="black")
    for left in (21, 49, 77):
        draw.rectangle((left, 70, left + 8, 82), fill="black")
    return image


def _make_vertical_left_sample() -> Image.Image:
    image = Image.new("RGB", (120, 150), "white")
    draw = ImageDraw.Draw(image)

    for top in (18, 54, 90):
        draw.rectangle((46, top, 80, top + 17), fill="black")
    for top in (21, 57, 93):
        draw.rectangle((26, top, 36, top + 8), fill="black")
    return image


def _make_horizontal_center_sample() -> Image.Image:
    image = Image.new("RGB", (140, 135), "white")
    draw = ImageDraw.Draw(image)

    for left in (18, 46, 74):
        draw.rectangle((left, 18, left + 17, 54), fill="black")
    for left in (21, 49, 77):
        draw.rectangle((left, 66, left + 8, 78), fill="black")
    for left in (18, 46, 74):
        draw.rectangle((left, 90, left + 17, 126), fill="black")
    return image


def _make_plain_sample() -> Image.Image:
    image = Image.new("RGB", (120, 90), "white")
    draw = ImageDraw.Draw(image)
    for left in (18, 48, 78):
        draw.rectangle((left, 30, left + 18, 68), fill="black")
    return image


def _make_partial_component_sample() -> Image.Image:
    image = Image.new("RGB", (160, 100), "white")
    draw = ImageDraw.Draw(image)

    for left in (16, 50, 84):
        draw.rectangle((left, 36, left + 19, 74), fill="black")
    draw.rectangle((118, 48, 124, 58), fill="black")
    return image


def _make_staggered_main_lines_sample() -> Image.Image:
    image = Image.new("RGB", (180, 110), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((12, 20, 30, 34), fill="black")
    draw.rectangle((108, 25, 126, 39), fill="black")
    draw.rectangle((42, 52, 60, 66), fill="black")
    draw.rectangle((138, 57, 156, 71), fill="black")
    return image


def _make_small_horizontal_sample() -> Image.Image:
    image = Image.new("RGB", (96, 76), "white")
    draw = ImageDraw.Draw(image)

    for left in (12, 34, 56):
        draw.rectangle((left, 34, left + 11, 58), fill="black")
    for left in (14, 36, 58):
        draw.rectangle((left, 18, left + 4, 26), fill="black")
    return image


def _make_large_vertical_sample() -> Image.Image:
    image = Image.new("RGB", (150, 220), "white")
    draw = ImageDraw.Draw(image)

    for top in (22, 82, 142):
        draw.rectangle((34, top, 94, top + 25), fill="black")
    for top in (26, 86, 146):
        draw.rectangle((106, top, 123, top + 11), fill="black")
    return image


def _make_overlapping_boundary_sample() -> Image.Image:
    image = Image.new("RGB", (150, 92), "white")
    draw = ImageDraw.Draw(image)

    for left in (16, 50, 84):
        draw.rectangle((left, 40, left + 19, 76), fill="black")
    for left in (4, 38, 72):
        draw.rectangle((left, 30, left + 8, 42), fill="black")
    return image


def _make_outline_heavy_boundary_sample() -> Image.Image:
    image = Image.new("RGB", (170, 110), "white")
    draw = ImageDraw.Draw(image)
    outline = (185, 185, 185)

    draw.rectangle((6, 24, 34, 84), fill=outline)
    draw.rectangle((10, 42, 30, 80), fill="black")

    draw.rectangle((58, 36, 108, 84), fill=outline)
    draw.rectangle((62, 40, 104, 80), fill="black")

    draw.rectangle((70, 12, 94, 30), fill=outline)
    draw.rectangle((74, 16, 90, 26), fill="black")
    return image


def _make_fragmented_furigana_sample() -> Image.Image:
    image = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((68, 8, 94, 12), fill="black")
    draw.rectangle((66, 18, 96, 30), fill="black")
    draw.rectangle((70, 34, 92, 36), fill="black")

    for left in (12, 54, 96):
        draw.rectangle((left, 44, left + 25, 92), fill="black")
    return image


def test_build_furigana_mask_removes_horizontal_furigana() -> None:
    image = _make_horizontal_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    assert result.selected_count == 3
    assert result.masked_pixel_count > 0
    assert any(line.role == "main" for line in result.lines)
    assert any(line.role == "furigana" for line in result.lines)
    furigana_line = next(line for line in result.lines if line.role == "furigana")
    assert furigana_line.left == 0
    assert furigana_line.right == image.width - 1
    assert result.image.getpixel((25, 30)) == (255, 255, 255)
    assert result.image.getpixel((40, 30)) == (255, 255, 255)
    assert result.image.getpixel((26, 60)) == (0, 0, 0)


def test_build_furigana_mask_removes_vertical_furigana() -> None:
    image = _make_vertical_sample()

    result = build_furigana_mask(image)

    assert result.orientation == VERTICAL
    assert result.selected_count == 3
    assert result.masked_pixel_count > 0
    furigana_line = next(line for line in result.lines if line.role == "furigana")
    assert furigana_line.top == 0
    assert furigana_line.bottom == image.height - 1
    assert result.image.getpixel((82, 25)) == (255, 255, 255)
    assert result.image.getpixel((45, 25)) == (0, 0, 0)


def test_build_furigana_mask_removes_bottom_horizontal_furigana() -> None:
    image = _make_horizontal_bottom_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    assert result.selected_count == 3
    assert result.image.getpixel((25, 75)) == (255, 255, 255)
    assert result.image.getpixel((26, 40)) == (0, 0, 0)


def test_build_furigana_mask_removes_left_vertical_furigana() -> None:
    image = _make_vertical_left_sample()

    result = build_furigana_mask(image)

    assert result.orientation == VERTICAL
    assert result.selected_count == 3
    assert result.image.getpixel((30, 25)) == (255, 255, 255)
    assert result.image.getpixel((55, 25)) == (0, 0, 0)


def test_build_furigana_mask_removes_center_horizontal_furigana() -> None:
    image = _make_horizontal_center_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    assert result.selected_count == 3
    assert result.image.getpixel((25, 72)) == (255, 255, 255)
    assert result.image.getpixel((26, 35)) == (0, 0, 0)
    assert result.image.getpixel((26, 108)) == (0, 0, 0)


def test_build_furigana_mask_leaves_plain_text_unchanged() -> None:
    image = _make_plain_sample()

    result = build_furigana_mask(image)

    assert result.selected_count == 0
    assert result.masked_pixel_count == 0
    assert np.array_equal(np.asarray(result.image), np.asarray(image))


def test_build_furigana_mask_does_not_mask_small_component_inside_main_line() -> None:
    image = _make_partial_component_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    assert result.selected_count == 0
    assert result.masked_pixel_count == 0
    assert result.image.getpixel((120, 52)) == (0, 0, 0)


def test_build_furigana_mask_splits_staggered_main_lines() -> None:
    image = _make_staggered_main_lines_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    main_lines = [line for line in result.lines if line.role == "main"]
    assert len(main_lines) >= 2
    assert main_lines[0].bottom < main_lines[-1].top
    assert all(line.left == 0 and line.right == image.width - 1 for line in result.lines)


def test_build_furigana_masks_keeps_horizontal_and_vertical_stats_separate() -> None:
    horizontal = _make_small_horizontal_sample()
    vertical = _make_large_vertical_sample()

    horizontal_result, vertical_result = build_furigana_masks([horizontal, vertical])

    assert horizontal_result.orientation == HORIZONTAL
    assert horizontal_result.selected_count == 3
    assert horizontal_result.image.getpixel((16, 20)) == (255, 255, 255)
    assert horizontal_result.image.getpixel((18, 40)) == (0, 0, 0)

    assert vertical_result.orientation == VERTICAL
    assert vertical_result.selected_count == 3
    assert vertical_result.image.getpixel((110, 30)) == (255, 255, 255)
    assert vertical_result.image.getpixel((46, 30)) == (0, 0, 0)


def test_build_furigana_mask_splits_overlapping_main_and_furigana_boundary() -> None:
    image = _make_overlapping_boundary_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    main_lines = [line for line in result.lines if line.role == "main"]
    furigana_lines = [line for line in result.lines if line.role == "furigana"]
    assert len(main_lines) == 1
    assert len(furigana_lines) == 1
    assert furigana_lines[0].bottom < main_lines[0].top
    assert result.selected_count > 0
    assert result.image.getpixel((8, 34)) == (255, 255, 255)
    assert result.image.getpixel((22, 56)) == (0, 0, 0)


def test_build_furigana_mask_ignores_outline_when_splitting_lines() -> None:
    image = _make_outline_heavy_boundary_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    main_lines = [line for line in result.lines if line.role == "main"]
    furigana_lines = [line for line in result.lines if line.role == "furigana"]
    assert len(main_lines) == 1
    assert len(furigana_lines) == 1
    assert furigana_lines[0].bottom < main_lines[0].top
    assert result.selected_count > 0
    assert result.image.getpixel((74, 13)) == (255, 255, 255)
    assert result.image.getpixel((16, 50)) == (0, 0, 0)


def test_build_furigana_mask_absorbs_thin_adjacent_furigana_fragments() -> None:
    image = _make_fragmented_furigana_sample()

    result = build_furigana_mask(image)

    assert result.orientation == HORIZONTAL
    assert sum(1 for line in result.lines if line.role == "furigana") >= 2
    assert result.image.getpixel((72, 10)) == (255, 255, 255)
    assert result.image.getpixel((72, 24)) == (255, 255, 255)
    assert result.image.getpixel((72, 35)) == (255, 255, 255)
    assert result.image.getpixel((20, 60)) == (0, 0, 0)
