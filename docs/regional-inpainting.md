# Multi-region inpainting

The img2img inpainting pipeline can combine the normal canvas/uploaded mask
with up to 64 additional region masks. The legacy `mask` request field keeps
its existing behavior when `inpaint_regions` is omitted.

## UI

Open `img2img -> Inpaint`, `Inpaint sketch`, or `Inpaint upload` and expand
`Multi-region inpaint`.

1. Draw the usual base mask, or upload a base mask.
2. Upload additional PNG/WebP/JPEG masks to `Additional region masks`.
3. Optionally enter comma-separated values for `Weights`, `Feather`, and
   `Latent noise seeds`. Values are matched to masks in upload order. Empty
   lists use `1.0`, `0`, and `-1` respectively.
4. Click `Preview merged mask` to inspect the combined mask before generation.

RGBA region masks use their alpha channel when it is not fully opaque.
Otherwise the luminance channel is used. Masks are resized to the source image
size.

## API

`POST /sdapi/v1/img2img` accepts the existing fields plus:

```json
{
  "init_images": ["data:image/png;base64,..."],
  "mask": "data:image/png;base64,...",
  "inpaint_regions": [
    {
      "mask": "data:image/png;base64,...",
      "weight": 1.0,
      "feather": 8,
      "seed": 1234,
      "name": "face"
    },
    {
      "mask": "data:image/png;base64,...",
      "weight": 0.7,
      "feather": 4,
      "seed": -1
    }
  ],
  "return_region_preview": true,
  "inpainting_fill": 2,
  "seed": 42,
  "prompt": "example",
  "steps": 20
}
```

Field rules:

- `mask` is optional when `inpaint_regions` is supplied.
- `weight` is in `[0, 1]`. A zero-weight region is ignored.
- `feather` is the per-region Gaussian feather radius in source-image pixels,
  in `[0, 256]`. The existing global `mask_blur` is applied to the merged mask
  afterwards.
- `seed` is `-1` or a non-negative integer. It controls masked-content latent
  noise only when `inpainting_fill` is `2` (`latent noise`). The normal image
  sampling seed remains the request-level `seed`.
- `return_region_preview` appends the rendered merged-mask preview to the
  response `images` list after generated/grid images.

Saving an inpaint result records region count, weights, feather values, and
seeds in the generation infotext.

Invalid region input returns HTTP `422` with the failing region index and
field, for example `Invalid inpaint_regions[1].mask`. UI preview errors are
shown in the status area and generation errors include the tab, region count,
weights, feather values, and seeds.

## Merge semantics

- The legacy mask is binarized and combined with the region masks as a union.
- At overlapping pixels the strongest `mask_alpha * weight` value wins.
- `inpainting_mask_invert` is applied after merging.
- Pixels with merged mask alpha `0` keep their exact source values through the
  existing latent blend and original-image overlay.
- Region masks are merged in memory and sampled once. The implementation does
  not launch one diffusion pass per region.
- On the test container, merging two feathered `512x512` masks takes about
  `4-5 ms`; this is below 10% of a typical diffusion sampling pass and is
  negligible compared with VAE/UNet work.

## Tests

```bash
python3 -m pytest -q test/test_regional_inpainting.py
```

The tests cover single-region and multi-region compositing, untouched pixels,
weighted overlap, feather transitions, deterministic region seed noise, empty
masks, and invalid-input messages.

The same fixture/mask path can be exercised end-to-end with a small checkpoint
through the existing WebUI API tests. Start the WebUI on the configured
`--api` endpoint first:

```bash
python3 -m pytest -q test/test_img2img.py
```

## Known limitations

- One prompt applies to all regions. Per-region prompts require separate
  sampling passes and are intentionally not part of this single-pass pipeline.
- Region seeds affect only `latent noise` masked-content initialization; they
  do not override the request-level sampler seed.
- Overlapping regions use the strongest weighted alpha rather than adding
  weights together.
- The preview is an approximate overlay intended for mask setup; generation
  still follows the resized and cropped image geometry configured in img2img.
