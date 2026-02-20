import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
from pathlib import Path
from diffusers.utils import export_to_gif


def bounding_rectangle(ori_img, mask):
    """
    Calculate the bounding rectangle of multiple rectangles.
    Args:
        rectangles (list of tuples): List of rectangles, where each rectangle is represented as (x, y, w, h)
    Returns:
        tuple: The bounding rectangle (x, y, w, h)
    """
    contours, _ = cv2.findContours(
        mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rectangles = [cv2.boundingRect(contour) for contour in contours]

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for x, y, w, h in rectangles:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    try:
        crop = ori_img[min_y:max_y, min_x:max_x]
        mask = mask[min_y:max_y, min_x:max_x]
    except:
        traceback.print_exc()
    return crop, mask


def prepare_input_image(
    input_image,
    input_mask,
    body_part_masks,
    body_part_keys,
    transform,
    device,
    weight_dtype,
    image_out_dir,
):

    ori_img = np.array(input_image)

    # Full-Body Mask
    if input_mask:
        ori_mask = (np.array(input_mask) / 255).astype(np.uint8)
        ori_mask = np.repeat(ori_mask[:, :, np.newaxis], 3, axis=2)

        # Masked Input Image
        ori_img = ori_img * ori_mask

    # Crop Full-Body Using Full-Body Mask
    # full_body_img, _ = bounding_rectangle(ori_img, ori_mask)
    full_body_img = ori_img
    full_body_pil = Image.fromarray(full_body_img)
    full_body_pil = ImageOps.pad(
        full_body_pil,
        (max(full_body_pil.size), max(full_body_pil.size)),
        color=(0, 0, 0),
    )
    full_body_pil = full_body_pil.resize((224, 224))

    # Save input images
    full_body_pil.save(image_out_dir / "full_body.png")

    full_body_image = (
        transform(full_body_pil, return_tensors="pt")
        .pixel_values.to(device)
        .to(weight_dtype)
    )

    # Crop Body Parts Using Body Part Masks
    body_part_images = {}
    for i in [1, 2, 3, 4]:
        if i not in body_part_keys:
            body_part_images[i] = torch.zeros_like(full_body_image).to(torch.uint8)
        else:
            # Body Part Mask
            body_part_mask = body_part_masks[i]
            body_part_mask = (np.array(body_part_mask) / 255).astype(np.uint8)
            body_part_mask = np.repeat(body_part_mask[:, :, np.newaxis], 3, axis=2)

            # Crop Body Part Using Body Part Mask
            body_part_img, _ = bounding_rectangle(ori_img, body_part_mask)
            body_part_pil = Image.fromarray(body_part_img)
            body_part_pil = ImageOps.pad(
                body_part_pil,
                (max(body_part_pil.size), max(body_part_pil.size)),
                color=(0, 0, 0),
            )
            body_part_pil = body_part_pil.resize((224, 224))

            # Save part images
            body_part_pil.save(image_out_dir / f"body_part_{i}.png")

            body_part_image = (
                transform(body_part_pil, return_tensors="pt")
                .pixel_values.to(device)
                .to(weight_dtype)
            )
            body_part_images[i] = body_part_image
    
    body_part_images_cat = torch.cat([body_part_images[k] for k in [1, 2, 3, 4]], dim=0)

    return torch.cat([full_body_image, body_part_images_cat], dim=0)

def prepare_input_image_tryon(
    head_image_pil, 
    head_mask, 
    top_image_pil, 
    top_mask,
    bottom_image_pil, 
    bottom_mask,
    shoes_image_pil,
    shoes_mask,
    transform,
    device,
    weight_dtype,
    image_out_dir,
):
   
    head_image = np.array(head_image_pil)
    head_mask = (np.array(head_mask) / 255).astype(np.uint8)
    head_mask = np.repeat(head_mask[:, :, np.newaxis], 3, axis=2)
    head_image = head_image * head_mask
    head_image, _ = bounding_rectangle(head_image, head_mask)
    head_image_pil = Image.fromarray(head_image)

    head_image_pil = ImageOps.pad(
        head_image_pil,
        (max(head_image_pil.size), max(head_image_pil.size)),
        color=(0, 0, 0),
    ).resize((224, 224))
    head_image_pil.save(image_out_dir / "head_image.png")

    top_image = np.array(top_image_pil)
    top_mask = (np.array(top_mask) / 255).astype(np.uint8)
    top_mask = np.repeat(top_mask[:, :, np.newaxis], 3, axis=2)
    top_image = top_image * top_mask
    top_image, _ = bounding_rectangle(top_image, top_mask)
    top_image_pil = Image.fromarray(top_image)

    top_image_pil = ImageOps.pad(
        top_image_pil,
        (max(top_image_pil.size), max(top_image_pil.size)),
        color=(0, 0, 0),
    ).resize((224, 224))
    top_image_pil.save(image_out_dir / "top_image.png")

    bottom_image = np.array(bottom_image_pil)
    bottom_mask = (np.array(bottom_mask) / 255).astype(np.uint8)
    bottom_mask = np.repeat(bottom_mask[:, :, np.newaxis], 3, axis=2)
    bottom_image = bottom_image * bottom_mask
    bottom_image, _ = bounding_rectangle(bottom_image, bottom_mask)
    bottom_image_pil = Image.fromarray(bottom_image)

    bottom_image_pil = ImageOps.pad(
        bottom_image_pil,
        (max(bottom_image_pil.size), max(bottom_image_pil.size)),
        color=(0, 0, 0),
    ).resize((224, 224))
    bottom_image_pil.save(image_out_dir / "bottom_image.png")

    shoes_image = np.array(shoes_image_pil)
    shoes_mask = (np.array(shoes_mask) / 255).astype(np.uint8)
    shoes_mask = np.repeat(shoes_mask[:, :, np.newaxis], 3, axis=2)
    shoes_image = shoes_image * shoes_mask
    shoes_image, _ = bounding_rectangle(shoes_image, shoes_mask)
    shoes_image_pil = Image.fromarray(shoes_image)

    shoes_image_pil = ImageOps.pad(
        shoes_image_pil,
        (max(shoes_image_pil.size), max(shoes_image_pil.size)),
        color=(0, 0, 0),
    ).resize((224, 224))
    shoes_image_pil.save(image_out_dir / "shoes_image.png")

    head_image = (
        transform(head_image_pil, return_tensors="pt")
        .pixel_values.to(device)
        .to(weight_dtype)
    )

    top_image = (
        transform(top_image_pil, return_tensors="pt")
        .pixel_values.to(device)
        .to(weight_dtype)
    )

    bottom_image = (
        transform(bottom_image_pil, return_tensors="pt")
        .pixel_values.to(device)
        .to(weight_dtype)
    )

    shoes_image = (
        transform(shoes_image_pil, return_tensors="pt")
        .pixel_values.to(device)
        .to(weight_dtype)
    )
    
    # No full body image
    full_body_image = torch.zeros_like(head_image).to(torch.uint8)

    return torch.cat([full_body_image, head_image, top_image, bottom_image, shoes_image], dim=0)


def generate_and_save_images(
    ip_model,
    input_image,
    cfg,
    prompt,
    seeds,
    image_out_dir,
    prompt_number,
    possible_keys,
):
    """Generate images based on the prompt and save them."""

    prompt_dir = image_out_dir / f"prompt_{prompt_number}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    batch_images = []

    # Generate images in pairs
    for i in range(cfg.num_images_per_prompt // 2):
        images = ip_model.generate(
            pil_image=input_image,
            num_samples=2,
            scale=cfg.adapter_attention_scale,
            num_inference_steps=cfg.num_inference_steps,
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.img_size,
            height=cfg.img_size,
            guidance_scale=cfg.guidance_scale,
            seed=seeds[2 * i : 2 * i + 2],
            possible_keys=possible_keys,
        )
        batch_images.extend(images)

    # Handle the case when the number of images is odd
    if cfg.num_images_per_prompt % 2 == 1:
        # Generate the last single image
        last_image = ip_model.generate(
            pil_image=input_image,
            num_samples=1,
            scale=cfg.adapter_attention_scale,
            num_inference_steps=cfg.num_inference_steps,
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.img_size,
            height=cfg.img_size,
            guidance_scale=cfg.guidance_scale,
            seed=[seeds[-1]],  # Use the last seed for the single image
            possible_keys=possible_keys,
        )
        batch_images.extend(last_image)

    # Save the images
    for idx, image in enumerate(batch_images):
        image.save(prompt_dir / f"image_{idx}.jpg")



def generate_and_save_images_controlnet(
    ip_model,
    input_image,
    openpose_image,
    control_scale,
    cfg,
    prompt,
    seeds,
    image_out_dir,
    prompt_number,
    ctrl_num,
    possible_keys
):
    """Generate images based on the prompt and save them."""

    prompt_dir = image_out_dir / f"prompt_{prompt_number}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    batch_images = []

    # Generate images in pairs
    for i in range(cfg.num_images_per_prompt // 2):
        images = ip_model.generate(
            pil_image=input_image,
            image=openpose_image,
            controlnet_conditioning_scale=control_scale,
            num_samples=2,
            scale=cfg.adapter_attention_scale,
            num_inference_steps=cfg.num_inference_steps,
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.img_size,
            height=cfg.img_size,
            guidance_scale=cfg.guidance_scale,
            seed=seeds[2 * i : 2 * i + 2],
            possible_keys=possible_keys,
        )
        batch_images.extend(images)

    # Handle the case when the number of images is odd
    if cfg.num_images_per_prompt % 2 == 1:
        # Generate the last single image
        last_image = ip_model.generate(
            pil_image=input_image,
            image=openpose_image,
            controlnet_conditioning_scale=control_scale,
            num_samples=1,
            scale=cfg.adapter_attention_scale,
            num_inference_steps=cfg.num_inference_steps,
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.img_size,
            height=cfg.img_size,
            guidance_scale=cfg.guidance_scale,
            seed=[seeds[-1]],  # Use the last seed for the single image
            possible_keys=possible_keys,
        )
        batch_images.extend(last_image)

    # Save the images
    for idx, image in enumerate(batch_images):
        control_path = os.path.join(prompt_dir, f"control_{ctrl_num}")

        if not os.path.exists(control_path):
            os.makedirs(control_path)

        image.save(Path(control_path) / f"image_{idx}_{cfg.seed}.jpg")
        

def generate_and_save_animation(
    ip_model,
    input_image,
    cfg,
    prompt,
    seeds,
    image_out_dir,
    prompt_number,
    possible_keys,
):

    batch_videos = []

    # Generate images in pairs
    for i in range(cfg.num_videos_per_prompt):
        video = ip_model.generate(
            pil_image=input_image,
            num_frames=cfg.num_frames,
            scale=cfg.adapter_attention_scale,
            num_inference_steps=cfg.num_inference_steps,
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.img_size,
            height=cfg.img_size,
            guidance_scale=cfg.guidance_scale,
            seed=seeds[i],
            possible_keys=possible_keys,
        )
        batch_videos.append(video)
        
    prompt_dir = image_out_dir / f"prompt_{prompt_number}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for idx, video in enumerate(batch_videos):
        output_dir = prompt_dir / f"video_{idx}.gif"
        export_to_gif(video, output_dir)

  