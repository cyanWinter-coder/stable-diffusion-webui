
import base64
import io

import numpy as np
import pytest
import requests
from PIL import Image


def decode_image(data_url):
    data = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def decode_mask(data_url):
    data = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("L")


@pytest.fixture()
def url_img2img(base_url):
    return f"{base_url}/sdapi/v1/img2img"


@pytest.fixture()
def simple_img2img_request(img2img_basic_image_base64):
    return {
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0.75,
        "eta": 0,
        "height": 64,
        "include_init_images": False,
        "init_images": [img2img_basic_image_base64],
        "inpaint_full_res": False,
        "inpaint_full_res_padding": 0,
        "inpainting_fill": 0,
        "inpainting_mask_invert": False,
        "mask": None,
        "mask_blur": 4,
        "n_iter": 1,
        "negative_prompt": "",
        "override_settings": {},
        "prompt": "example prompt",
        "resize_mode": 0,
        "restore_faces": False,
        "s_churn": 0,
        "s_noise": 1,
        "s_tmax": 0,
        "s_tmin": 0,
        "sampler_index": "Euler a",
        "seed": -1,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "steps": 3,
        "styles": [],
        "subseed": -1,
        "subseed_strength": 0,
        "tiling": False,
        "width": 64,
    }


def test_img2img_simple_performed(url_img2img, simple_img2img_request):
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_inpainting_masked_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["mask"] = mask_basic_image_base64
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_inpainting_with_inverted_masked_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["mask"] = mask_basic_image_base64
    simple_img2img_request["inpainting_mask_invert"] = True
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_inpainting_single_region_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["inpaint_regions"] = [
        {"mask": mask_basic_image_base64, "weight": 1.0, "feather": 2, "seed": 1234},
    ]
    response = requests.post(url_img2img, json=simple_img2img_request)
    assert response.status_code == 200

    source = decode_image(simple_img2img_request["init_images"][0])
    generated = decode_image(response.json()["images"][0])
    mask = np.asarray(decode_mask(mask_basic_image_base64))
    assert np.array_equal(np.asarray(generated)[mask == 0], np.asarray(source)[mask == 0])


def test_inpainting_multiple_regions_with_preview_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["inpaint_regions"] = [
        {"mask": mask_basic_image_base64, "weight": 1.0, "feather": 2, "seed": 1234},
        {"mask": mask_basic_image_base64, "weight": 0.35, "feather": 4, "seed": 5678},
    ]
    simple_img2img_request["return_region_preview"] = True
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_inpainting_seed_reproducible(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request.update({
        "seed": 4242,
        "inpainting_fill": 2,
        "inpaint_regions": [
            {"mask": mask_basic_image_base64, "weight": 0.6, "feather": 2, "seed": 99},
        ],
    })
    first = requests.post(url_img2img, json=simple_img2img_request)
    second = requests.post(url_img2img, json=simple_img2img_request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["images"] == second.json()["images"]


def test_inpainting_empty_region_mask_performed(url_img2img, simple_img2img_request, empty_mask_image_base64):
    simple_img2img_request["inpaint_regions"] = [
        {"mask": empty_mask_image_base64, "weight": 1.0, "feather": 4, "seed": -1},
    ]
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_inpainting_invalid_region_reports_missing_mask(url_img2img, simple_img2img_request):
    simple_img2img_request["inpaint_regions"] = [{"weight": 0.5}]
    response = requests.post(url_img2img, json=simple_img2img_request)

    assert response.status_code == 422
    assert "inpaint_regions[0].mask" in response.json()["detail"]


def test_img2img_sd_upscale_performed(url_img2img, simple_img2img_request):
    simple_img2img_request["script_name"] = "sd upscale"
    simple_img2img_request["script_args"] = ["", 8, "Lanczos", 2.0]
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200
