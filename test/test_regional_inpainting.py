import numpy as np
import pytest
from PIL import Image, ImageDraw

from modules import regional_inpainting


def make_image(size=(96, 72)):
    x = np.linspace(0, 255, size[0], dtype=np.uint8)
    y = np.linspace(0, 255, size[1], dtype=np.uint8)
    red = np.tile(x, (size[1], 1))
    green = np.tile(y[:, None], (1, size[0]))
    blue = np.full((size[1], size[0]), 40, dtype=np.uint8)
    return Image.fromarray(np.dstack((red, green, blue)), mode="RGB")


def make_box_mask(size, box, fill=255):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=fill)
    return mask


def composite(original, merged_mask, color=(250, 30, 10)):
    generated = Image.new("RGB", original.size, color)
    return np.asarray(regional_inpainting.composite_generated_image(original, generated, merged_mask))


def test_single_region_keeps_unselected_pixels_exact():
    original = make_image()
    mask = make_box_mask(original.size, (10, 12, 40, 42))
    regions = [regional_inpainting.InpaintRegion(mask=mask, weight=1.0, seed=11)]

    merged, normalized = regional_inpainting.merge_region_masks(None, regions, size=original.size)
    output = composite(original, merged)

    alpha = np.asarray(merged)
    source = np.asarray(original)
    assert len(normalized) == 1
    assert np.array_equal(output[alpha == 0], source[alpha == 0])
    assert np.array_equal(output[20, 20], np.asarray([250, 30, 10], dtype=np.uint8))


def test_multiple_regions_merge_weights_and_keep_other_pixels_exact():
    original = make_image()
    first = make_box_mask(original.size, (4, 4, 30, 30))
    second = make_box_mask(original.size, (58, 30, 90, 68))
    regions = [
        regional_inpainting.InpaintRegion(mask=first, weight=1.0),
        regional_inpainting.InpaintRegion(mask=second, weight=0.5),
    ]

    merged, normalized = regional_inpainting.merge_region_masks(None, regions, size=original.size)
    alpha = np.asarray(merged)
    output = composite(original, merged)
    source = np.asarray(original)

    assert len(normalized) == 2
    assert alpha[15, 15] == 255
    assert 100 <= alpha[50, 70] <= 140
    assert np.array_equal(output[alpha == 0], source[alpha == 0])


def test_overlapping_regions_do_not_exceed_full_mask_weight():
    size = (32, 32)
    first = make_box_mask(size, (4, 4, 20, 20))
    second = make_box_mask(size, (10, 10, 28, 28))
    merged, _ = regional_inpainting.merge_region_masks(
        None,
        [
            regional_inpainting.InpaintRegion(mask=first, weight=0.4),
            regional_inpainting.InpaintRegion(mask=second, weight=0.8),
        ],
        size=size,
    )

    assert np.asarray(merged)[15, 15] == 204
    assert int(np.asarray(merged).max()) == 204


def test_weighted_mask_is_resized_without_losing_soft_alpha():
    source_size = (64, 64)
    small_mask = make_box_mask((32, 32), (4, 4, 27, 27))
    merged, _ = regional_inpainting.merge_region_masks(
        None,
        [regional_inpainting.InpaintRegion(mask=small_mask, weight=0.5)],
        size=source_size,
    )

    resized = regional_inpainting.resize_mask(merged, (8, 8))
    values = np.asarray(resized)
    assert merged.size == source_size
    assert np.any((values > 0) & (values < 255))


def test_weighted_overlay_blends_generated_and_original_pixels():
    original = Image.new("RGB", (32, 32), (20, 20, 20))
    generated = Image.new("RGB", (32, 32), (220, 220, 220))
    mask = Image.new("L", (32, 32), 128)

    output = np.asarray(regional_inpainting.composite_generated_image(original, generated, mask))
    assert output.shape == (32, 32, 3)
    assert np.all((output > 100) & (output < 150))


def test_feather_creates_progressive_soft_edge_without_a_hard_seam():
    size = (80, 80)
    mask = make_box_mask(size, (20, 20, 59, 59))
    regions = [regional_inpainting.InpaintRegion(mask=mask, weight=1.0, feather=4.0)]

    merged, _ = regional_inpainting.merge_region_masks(None, regions, size=size)
    row = np.asarray(merged)[40, 10:55]
    transition = row[(row > 0) & (row < 255)]

    assert len(transition) >= 6
    assert int(row[0]) <= 2
    assert int(row.max()) == 255
    leading_edge = row[: np.argmax(row) + 1]
    assert np.all(np.diff(leading_edge.astype(np.int16)) >= -3)
    assert np.any((row > 40) & (row < 215))


def test_region_seed_noise_is_reproducible_and_seed_dependent():
    first = regional_inpainting.seeded_latent_noise(123456, (4, 8, 8))
    same = regional_inpainting.seeded_latent_noise(123456, (4, 8, 8))
    different = regional_inpainting.seeded_latent_noise(123457, (4, 8, 8))

    assert np.array_equal(first, same)
    assert not np.array_equal(first, different)
    assert first.shape == (4, 8, 8)


def test_empty_mask_is_a_noop_and_preview_does_not_raise():
    original = make_image()
    empty = Image.new("L", original.size, 0)

    merged, regions = regional_inpainting.merge_region_masks(empty, [], size=original.size)
    output = composite(original, merged)
    preview = regional_inpainting.render_preview(original, merged)

    assert regions == []
    assert int(np.asarray(merged).max()) == 0
    assert np.array_equal(output, np.asarray(original))
    assert preview.size == original.size
    assert np.array_equal(np.asarray(preview), np.asarray(original))


def test_invalid_region_input_reports_region_and_expected_value():
    with pytest.raises(regional_inpainting.RegionInputError, match=r"weight"):
        regional_inpainting.InpaintRegion(mask=Image.new("L", (8, 8), 255), weight=1.5)

    with pytest.raises(regional_inpainting.RegionInputError, match=r"inpaint region 0.*missing mask"):
        regional_inpainting.normalize_regions([{"weight": 0.5}])

    with pytest.raises(regional_inpainting.RegionInputError, match=r"inpaint region 0.*weight"):
        regional_inpainting.normalize_regions([{"mask": Image.new("L", (8, 8), 255), "weight": 1.5}])

    with pytest.raises(regional_inpainting.RegionInputError, match=r"2 value\(s\).*3 uploaded"):
        regional_inpainting.parse_region_values("1.0, 0.5", 3, 1.0, "region weights")
