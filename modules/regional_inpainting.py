from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


MAX_REGIONS = 64
MAX_FEATHER = 256.0


class RegionInputError(ValueError):
    """Raised when multi-region inpainting input cannot be used."""


@dataclass
class InpaintRegion:
    """A single masked regeneration region.

    ``weight`` scales the mask alpha, ``feather`` adds a Gaussian falloff in
    source-image pixels, and ``seed`` controls the latent-noise fill for this
    region when ``inpainting_fill == 2``.  A seed of ``None`` or ``-1`` uses
    the request's normal seeded noise.
    """

    mask: Image.Image
    weight: float = 1.0
    feather: float = 0.0
    seed: int | None = None
    name: str | None = None

    def __post_init__(self):
        if isinstance(self.mask, np.ndarray):
            try:
                self.mask = Image.fromarray(self.mask)
            except Exception as exc:
                raise RegionInputError(f"Invalid inpaint region mask array with shape {self.mask.shape}: {exc}") from exc
        if not isinstance(self.mask, Image.Image):
            raise RegionInputError(
                f"Invalid inpaint region mask: expected a PIL image, got {type(self.mask).__name__}."
            )
        if self.mask.width <= 0 or self.mask.height <= 0:
            raise RegionInputError(f"Invalid inpaint region mask size: {self.mask.size}.")

        try:
            self.weight = float(self.weight)
        except (TypeError, ValueError) as exc:
            raise RegionInputError(f"Invalid inpaint region weight {self.weight!r}: expected a number from 0 to 1.") from exc
        if not np.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise RegionInputError(f"Invalid inpaint region weight {self.weight!r}: expected a number from 0 to 1.")

        try:
            self.feather = float(self.feather)
        except (TypeError, ValueError) as exc:
            raise RegionInputError(f"Invalid inpaint region feather {self.feather!r}: expected pixels from 0 to {MAX_FEATHER:g}.") from exc
        if not np.isfinite(self.feather) or not 0.0 <= self.feather <= MAX_FEATHER:
            raise RegionInputError(
                f"Invalid inpaint region feather {self.feather!r}: expected pixels from 0 to {MAX_FEATHER:g}."
            )

        if self.seed is not None:
            try:
                raw_seed = self.seed
                self.seed = int(raw_seed)
            except (TypeError, ValueError) as exc:
                raise RegionInputError(f"Invalid inpaint region seed {self.seed!r}: expected an integer or -1.") from exc
            if isinstance(raw_seed, float) and raw_seed != self.seed:
                raise RegionInputError(f"Invalid inpaint region seed {raw_seed!r}: expected an integer or -1.")
            if self.seed < -1:
                raise RegionInputError(f"Invalid inpaint region seed {self.seed}: expected -1 or a non-negative integer.")


def normalize_regions(regions: Iterable[Any] | None) -> list[InpaintRegion]:
    """Validate API/UI region values without mutating the caller's objects."""

    if regions is None:
        return []
    if isinstance(regions, (str, bytes, Mapping)) or not isinstance(regions, Sequence):
        raise RegionInputError(
            "Invalid inpaint_regions: expected a list of region objects, each with a mask, weight, feather and seed."
        )
    if len(regions) > MAX_REGIONS:
        raise RegionInputError(f"Invalid inpaint_regions: got {len(regions)} regions, maximum is {MAX_REGIONS}.")

    normalized = []
    for index, region in enumerate(regions):
        if isinstance(region, InpaintRegion):
            try:
                normalized.append(
                    InpaintRegion(
                        mask=region.mask,
                        weight=region.weight,
                        feather=region.feather,
                        seed=region.seed,
                        name=region.name,
                    )
                )
            except RegionInputError as exc:
                raise RegionInputError(f"Invalid inpaint region {index}: {exc}") from exc
            continue
        if isinstance(region, Image.Image):
            normalized.append(InpaintRegion(mask=region))
            continue
        if not isinstance(region, Mapping):
            raise RegionInputError(
                f"Invalid inpaint region {index}: expected an object with a mask, got {type(region).__name__}."
            )

        mask = region.get("mask", region.get("mask_image", region.get("image")))
        if mask is None:
            noise_seed = region.get("seed")
            raise RegionInputError(
                f"Invalid inpaint region {index}: missing mask (weight={region.get('weight', 1.0)!r}, seed={noise_seed!r})."
            )
        try:
            normalized.append(
                InpaintRegion(
                    mask=mask,
                    weight=region.get("weight", region.get("region_weight", 1.0)),
                    feather=region.get("feather", region.get("mask_feather", region.get("blur", 0.0))),
                    seed=region.get("seed", region.get("region_seed")),
                    name=region.get("name", region.get("label")),
                )
            )
        except RegionInputError as exc:
            raise RegionInputError(f"Invalid inpaint region {index}: {exc}") from exc
    return normalized


def mask_to_grayscale(mask: Image.Image) -> Image.Image:
    """Return a soft L mask, preferring alpha for RGBA masks."""

    if not isinstance(mask, Image.Image):
        raise RegionInputError(f"Invalid mask: expected a PIL image, got {type(mask).__name__}.")

    if mask.mode in ("RGBA", "LA") and mask.getchannel("A").getextrema() != (255, 255):
        return mask.getchannel("A").convert("L")
    return mask.convert("L")


def binarize_mask(mask: Image.Image, round: bool = True) -> Image.Image:
    """Match the legacy img2img mask conversion for the primary mask."""

    if mask.mode == "RGBA" and mask.getextrema()[-1] != (255, 255):
        if round:
            return mask.getchannel("A").convert("L").point(lambda value: 255 if value > 128 else 0)
        return mask.getchannel("A").convert("L")
    return mask.convert("L")


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if mask.size == size:
        return mask
    return mask.resize(size, resample=Image.Resampling.LANCZOS)


def feather_mask(mask: Image.Image, feather: float) -> Image.Image:
    if feather <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=feather))


def merge_region_masks(
    base_mask: Image.Image | None,
    regions: Iterable[Any] | None,
    size: tuple[int, int] | None = None,
    invert: bool = False,
    round_base: bool = True,
) -> tuple[Image.Image, list[InpaintRegion]]:
    """Merge a legacy mask and weighted region masks into one soft L mask.

    The legacy mask is binarized exactly like the old img2img path.  Regional
    masks keep their grayscale/alpha softness; feather is applied per region
    before compositing.  At overlapping pixels the strongest weighted region
    wins, so weights cannot create values above fully selected.
    """

    normalized = normalize_regions(regions)
    if size is None:
        if base_mask is not None:
            size = base_mask.size
        elif normalized:
            size = normalized[0].mask.size
        else:
            raise RegionInputError("Cannot merge inpainting regions without a mask or an explicit image size.")

    if base_mask is None:
        combined = np.zeros((size[1], size[0]), dtype=np.float32)
    else:
        base = binarize_mask(base_mask, round=round_base)
        base = resize_mask(base, size)
        combined = np.asarray(base, dtype=np.float32) / 255.0

    for index, region in enumerate(normalized):
        if region.weight == 0:
            continue
        try:
            region_mask = resize_mask(mask_to_grayscale(region.mask), size)
            region_mask = feather_mask(region_mask, region.feather)
        except Exception as exc:
            raise RegionInputError(
                f"Failed to prepare inpaint region {index} (weight={region.weight}, feather={region.feather}, "
                f"mask_size={region.mask.size}, target_size={size})."
            ) from exc

        alpha = np.asarray(region_mask, dtype=np.float32) / 255.0
        np.maximum(combined, alpha * region.weight, out=combined)

    if invert:
        combined = 1.0 - combined

    merged = np.clip(np.rint(combined * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(merged, mode="L"), normalized


def render_preview(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (255, 64, 64),
    opacity: float = 0.45,
) -> Image.Image:
    """Render a tinted mask and its boundary over the source image."""

    if not isinstance(image, Image.Image):
        raise RegionInputError(f"Invalid preview image: expected a PIL image, got {type(image).__name__}.")
    if not isinstance(mask, Image.Image):
        raise RegionInputError(f"Invalid preview mask: expected a PIL image, got {type(mask).__name__}.")
    if not 0.0 <= float(opacity) <= 1.0:
        raise RegionInputError(f"Invalid preview opacity {opacity!r}: expected a number from 0 to 1.")

    base = image.convert("RGB")
    mask_l = resize_mask(mask_to_grayscale(mask), base.size)
    alpha = np.asarray(mask_l, dtype=np.float32)[..., None] / 255.0
    base_array = np.asarray(base, dtype=np.float32)
    tint = np.asarray(color, dtype=np.float32)[None, None, :]
    preview = base_array * (1.0 - alpha * float(opacity)) + tint * (alpha * float(opacity))

    edge = mask_l.filter(ImageFilter.FIND_EDGES)
    edge_alpha = np.asarray(edge, dtype=np.float32)[..., None] / 255.0
    edge_alpha *= np.asarray(mask_l, dtype=np.float32)[..., None] > 0
    preview = preview * (1.0 - edge_alpha) + np.asarray((255, 255, 255), dtype=np.float32)[None, None, :] * edge_alpha

    return Image.fromarray(np.clip(np.rint(preview), 0, 255).astype(np.uint8), mode="RGB")


def composite_generated_image(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Composite generated pixels using mask alpha; zero-alpha pixels stay exact."""

    if original.size != generated.size:
        generated = generated.resize(original.size, resample=Image.Resampling.LANCZOS)
    original_rgb = original.convert("RGB")
    generated_rgb = generated.convert("RGB")
    alpha = resize_mask(mask_to_grayscale(mask), original_rgb.size)
    return Image.composite(generated_rgb, original_rgb, alpha)


def seeded_latent_noise(seed: int, shape: Sequence[int]) -> np.ndarray:
    """Create reproducible latent noise for a region seed."""

    if int(seed) < -1:
        raise RegionInputError(f"Invalid region seed {seed}: expected -1 or a non-negative integer.")
    resolved_seed = max(int(seed), 0)
    random = np.random.RandomState(resolved_seed & 0xFFFFFFFF)
    return random.standard_normal(tuple(shape)).astype(np.float32)


def parse_region_values(text: str | None, count: int, default: Any, label: str) -> list[Any]:
    """Parse comma-separated UI values and validate their count."""

    if not text or not text.strip():
        return [default for _ in range(count)]

    raw_values = [value.strip() for value in text.replace(";", ",").split(",") if value.strip()]
    if len(raw_values) != count:
        raise RegionInputError(
            f"Invalid {label}: got {len(raw_values)} value(s) for {count} uploaded region mask(s): {text!r}."
        )
    return raw_values


def describe_regions(regions: Sequence[InpaintRegion]) -> dict[str, str]:
    """Return stable infotext fields for a regional request."""

    return {
        "Regional inpaint regions": str(len(regions)),
        "Regional inpaint weights": ", ".join(f"{region.weight:g}" for region in regions),
        "Regional inpaint feather": ", ".join(f"{region.feather:g}" for region in regions),
        "Regional inpaint seeds": ", ".join("-1" if region.seed is None else str(region.seed) for region in regions),
    }
