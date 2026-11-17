import logging
import os
from contextlib import closing
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
import gradio as gr

from modules import images, regional_inpainting
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
from modules.sd_models import get_closet_checkpoint_match
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.scripts


def _region_file_path(file):
    if isinstance(file, str):
        return file
    path = getattr(file, "name", None)
    if path:
        return path
    raise regional_inpainting.RegionInputError(
        f"Invalid uploaded region mask {file!r}: expected a file path from the UI."
    )


def load_inpaint_regions(region_files, region_weights, region_feathers, region_seeds):
    """Build validated regions from the img2img UI controls."""

    files = list(region_files or [])
    weights = regional_inpainting.parse_region_values(region_weights, len(files), 1.0, "region weights")
    feathers = regional_inpainting.parse_region_values(region_feathers, len(files), 0.0, "region feather values")
    seeds = regional_inpainting.parse_region_values(region_seeds, len(files), -1, "region seeds")

    regions = []
    for index, (file, weight, feather, seed) in enumerate(zip(files, weights, feathers, seeds)):
        path = _region_file_path(file)
        try:
            mask = images.read(path)
        except Exception as exc:
            raise regional_inpainting.RegionInputError(
                f"Failed to read inpaint region {index} mask {path!r}: {exc}"
            ) from exc
        try:
            regions.append(
                regional_inpainting.InpaintRegion(
                    mask=mask,
                    weight=float(weight),
                    feather=float(feather),
                    seed=int(seed),
                    name=os.path.basename(path),
                )
            )
        except (TypeError, ValueError) as exc:
            raise regional_inpainting.RegionInputError(
                f"Invalid inpaint region {index} from {path!r} "
                f"(weight={weight!r}, feather={feather!r}, seed={seed!r}): {exc}"
            ) from exc
    return regions


def _inpaint_preview_source(mode, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, mask_alpha, init_img_inpaint, init_mask_inpaint):
    mode = int(mode) if mode is not None else 0
    if mode == 2 and isinstance(init_img_with_mask, dict):
        return init_img_with_mask.get("image"), init_img_with_mask.get("mask")
    if mode == 3:
        image = inpaint_color_sketch
        original = inpaint_color_sketch_orig or inpaint_color_sketch
        if image is None or original is None:
            return image, None
        difference = np.any(np.array(image) != np.array(original), axis=-1)
        mask = Image.fromarray(difference.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - float(mask_alpha or 0) / 100)
        return image, mask
    if mode == 4:
        return init_img_inpaint, init_mask_inpaint
    if mode in (0, 1):
        return init_img if mode == 0 else sketch, None
    raise regional_inpainting.RegionInputError(
        f"Cannot preview inpainting regions for tab {mode}: select Inpaint, Inpaint sketch or Inpaint upload."
    )


def preview_inpaint_regions(mode, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, mask_blur, mask_alpha, inpainting_mask_invert, init_img_inpaint, init_mask_inpaint, region_files, region_weights, region_feathers, region_seeds):
    """Gradio callback for the multi-region mask preview."""

    try:
        image, mask = _inpaint_preview_source(
            mode,
            init_img,
            sketch,
            init_img_with_mask,
            inpaint_color_sketch,
            inpaint_color_sketch_orig,
            mask_alpha,
            init_img_inpaint,
            init_mask_inpaint,
        )
        if image is None:
            raise regional_inpainting.RegionInputError(
                "Cannot preview inpainting regions: no source image is selected."
            )
        regions = load_inpaint_regions(region_files, region_weights, region_feathers, region_seeds)
        if mask is None and not regions:
            raise regional_inpainting.RegionInputError(
                "Cannot preview inpainting regions: no mask was drawn or uploaded."
            )

        merged, regions = regional_inpainting.merge_region_masks(
            mask,
            regions,
            size=image.size,
            invert=bool(inpainting_mask_invert),
        )
        if mask_blur and float(mask_blur) > 0:
            merged = merged.filter(ImageFilter.GaussianBlur(radius=float(mask_blur)))
        preview = regional_inpainting.render_preview(image, merged)
        return preview, f"Previewing {len(regions)} extra region(s); legacy mask: {'yes' if mask is not None else 'no'}."
    except regional_inpainting.RegionInputError as exc:
        return None, escape(f"Region preview failed. Inputs: tab={mode!r}, region_count={len(region_files or [])}. Details: {exc}")
    except Exception as exc:
        logging.exception("Unexpected failure while previewing multi-region inpaint masks.")
        return None, escape(f"Region preview failed at preview_inpaint_regions. Inputs: tab={mode!r}, region_count={len(region_files or [])}. Details: {exc}")


def process_batch(p, input, output_dir, inpaint_mask_dir, args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    output_dir = output_dir.strip()
    processing.fix_seed(p)

    if isinstance(input, str):
        batch_images = list(shared.walk_files(input, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")))
    else:
        batch_images = [os.path.abspath(x.name) for x in input]

    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks)

        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")

    print(f"Will process {len(batch_images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    state.job_count = len(batch_images) * p.n_iter

    # extract "default" params to use in case getting png info fails
    prompt = p.prompt
    negative_prompt = p.negative_prompt
    seed = p.seed
    cfg_scale = p.cfg_scale
    sampler_name = p.sampler_name
    steps = p.steps
    override_settings = p.override_settings
    sd_model_checkpoint_override = get_closet_checkpoint_match(override_settings.get("sd_model_checkpoint", None))
    batch_results = None
    discard_further_results = False
    for i, image in enumerate(batch_images):
        state.job = f"{i+1} out of {len(batch_images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted or state.stopping_generation:
            break

        try:
            img = images.read(image)
        except UnidentifiedImageError as e:
            print(e)
            continue
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)

        if to_scale:
            p.width = int(img.width * scale_by)
            p.height = int(img.height * scale_by)

        p.init_images = [img] * p.batch_size

        image_path = Path(image)
        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
            if len(inpaint_masks) == 1:
                mask_image_path = inpaint_masks[0]
            else:
                # try to find corresponding mask for an image using simple filename matching
                mask_image_dir = Path(inpaint_mask_dir)
                masks_found = list(mask_image_dir.glob(f"{image_path.stem}.*"))

                if len(masks_found) == 0:
                    print(f"Warning: mask is not found for {image_path} in {mask_image_dir}. Skipping it.")
                    continue

                # it should contain only 1 matching mask
                # otherwise user has many masks with the same name but different extensions
                mask_image_path = masks_found[0]

            mask_image = images.read(mask_image_path)
            p.image_mask = mask_image

        if use_png_info:
            try:
                info_img = img
                if png_info_dir:
                    info_img_path = os.path.join(png_info_dir, os.path.basename(image))
                    info_img = images.read(info_img_path)
                geninfo, _ = images.read_info_from_image(info_img)
                parsed_parameters = parse_generation_parameters(geninfo)
                parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or {})}
            except Exception:
                parsed_parameters = {}

            p.prompt = prompt + (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else "")
            p.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
            p.seed = int(parsed_parameters.get("Seed", seed))
            p.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
            p.sampler_name = parsed_parameters.get("Sampler", sampler_name)
            p.steps = int(parsed_parameters.get("Steps", steps))

            model_info = get_closet_checkpoint_match(parsed_parameters.get("Model hash", None))
            if model_info is not None:
                p.override_settings['sd_model_checkpoint'] = model_info.name
            elif sd_model_checkpoint_override:
                p.override_settings['sd_model_checkpoint'] = sd_model_checkpoint_override
            else:
                p.override_settings.pop("sd_model_checkpoint", None)

        if output_dir:
            p.outpath_samples = output_dir
            p.override_settings['save_to_dirs'] = False
            p.override_settings['save_images_replace_action'] = "Add number suffix"
            if p.n_iter > 1 or p.batch_size > 1:
                p.override_settings['samples_filename_pattern'] = f'{image_path.stem}-[generation_number]'
            else:
                p.override_settings['samples_filename_pattern'] = f'{image_path.stem}'

        proc = modules.scripts.scripts_img2img.run(p, *args)

        if proc is None:
            p.override_settings.pop('save_images_replace_action', None)
            proc = process_images(p)

        if not discard_further_results and proc:
            if batch_results:
                batch_results.images.extend(proc.images)
                batch_results.infotexts.extend(proc.infotexts)
            else:
                batch_results = proc

            if 0 <= shared.opts.img2img_batch_show_results_limit < len(batch_results.images):
                discard_further_results = True
                batch_results.images = batch_results.images[:int(shared.opts.img2img_batch_show_results_limit)]
                batch_results.infotexts = batch_results.infotexts[:int(shared.opts.img2img_batch_show_results_limit)]

    return batch_results


def img2img(id_task: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, img2img_batch_source_type: str, img2img_batch_upload: list, region_files=None, region_weights=None, region_feathers=None, region_seeds=None, *args):
    override_settings = create_override_settings_dict(override_settings_texts)

    is_batch = mode == 5

    if mode == 0:  # img2img
        image = init_img
        mask = None
    elif mode == 1:  # img2img sketch
        image = sketch
        mask = None
    elif mode == 2:  # inpaint
        image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
        mask = processing.create_binary_mask(mask)
    elif mode == 3:  # inpaint sketch
        image = inpaint_color_sketch
        orig = inpaint_color_sketch_orig or inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
        blur = ImageFilter.GaussianBlur(mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
    elif mode == 4:  # inpaint upload mask
        image = init_img_inpaint
        mask = init_mask_inpaint
    else:
        image = None
        mask = None

    image = images.fix_image(image)
    mask = images.fix_image(mask)

    if region_files or region_weights or region_feathers or region_seeds:
        try:
            inpaint_regions = load_inpaint_regions(region_files, region_weights, region_feathers, region_seeds)
        except regional_inpainting.RegionInputError as exc:
            raise ValueError(
                f"Invalid multi-region inpainting request in img2img (mode={mode}, "
                f"region_count={len(region_files or [])}, weights={region_weights!r}, "
                f"feather={region_feathers!r}, seeds={region_seeds!r}). Details: {exc}"
            ) from exc
    else:
        inpaint_regions = None

    if selected_scale_tab == 1 and not is_batch:
        assert image, "Can't scale by because no image is selected"

        width = int(image.width * scale_by)
        height = int(image.height * scale_by)

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        inpaint_regions=inpaint_regions,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_img2img
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    with closing(p):
        if is_batch:
            if img2img_batch_source_type == "upload":
                assert isinstance(img2img_batch_upload, list) and img2img_batch_upload
                output_dir = ""
                inpaint_mask_dir = ""
                png_info_dir = img2img_batch_png_info_dir if not shared.cmd_opts.hide_ui_dir_config else ""
                processed = process_batch(p, img2img_batch_upload, output_dir, inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=png_info_dir)
            else: # "from dir"
                assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
                processed = process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=img2img_batch_png_info_dir)

            if processed is None:
                processed = Processed(p, [], p.seed, "")
        else:
            processed = modules.scripts.scripts_img2img.run(p, *args)
            if processed is None:
                processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
