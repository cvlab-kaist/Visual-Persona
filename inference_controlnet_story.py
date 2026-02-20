from tqdm import tqdm
import json
import pyrallis
import torch
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageOps
from typing import Optional, Tuple, Dict
from PIL import Image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
from controlnet_aux import OpenposeDetector
from transformers import CLIPImageProcessor
from model.visual_persona.vp_test import Visual_Persona
from model.model_utils import import_model_class_from_model_name_or_path

from inference_utils.utils import (
    prepare_input_image,
    generate_and_save_images,
    generate_and_save_images_controlnet
)


@dataclass
class Config:
    # Path to the input images
    inputs_path: Path = Path("data_inference/evaluation_controlnet/images")
    full_mask_path: Path = Path("data_inference/evaluation_controlnet/full_body_masks")
    part_mask_path: Path = Path("data_inference/evaluation_controlnet/body_part_masks")
    out_dir: Path = Path("./results/controlnet_story")

    control_paths = Path("data_inference/control_images_story")

    # Configuration settings
    use_dino: bool = True  # True, False
    customize_prompt: bool = True  # True, False
    decoder_type: str = "Transformer"  # Linear, Transformers

    # Model paths
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path: Optional[str] = "madebyollin/sdxl-vae-fp16-fix"
    encoder_ckpt: str = (
        "pretrained_models/dinov2_vitg14_pretrain.pth"
        if use_dino
        else "pretrained_models/image_encoder"
    )
    decoder_ckpt: str = (
        "pretrained_models/weight.bin"
    )
    pretrained_controlnet_name_or_path: str = "thibaud/controlnet-openpose-sdxl-1.0"
    pretrained_openpose_name_or_path: str = "lllyasviel/ControlNet"

    # Prompt settings
    with open("data_inference/control_images_story/control_prompts.txt", "r") as f:
        prompts = f.read().split("\n")

    # Inference settings
    device: str = "cuda:0"
    seed: int = 77
    control_scale: float = 1.0
    adapter_attention_scale: float = 0.7
    num_inference_steps: int = 50
    guidance_scale: float = 8
    num_images_per_prompt: int = 1
    img_size: int = 1024
    visual_persona_tokens: int = 256
    mixed_precision: str = "fp16"
    use_freeu: bool = False
    revision: Optional[str] = None
    variant: Optional[str] = None
    negative_prompt: str = (
        "low quality, bad quality, NSFW, ugly, disfigured, deformed, three arms, three legs, fused fingers, too many fingers, multiple people, cloned face, bad proportions, monochrome, lowres, bad anatomy, worst quality"
    )

def init_pipeline(cfg: Config) -> Tuple[StableDiffusionXLPipeline, Dict]:

    # Import model classes from the specified paths
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path, cfg.revision, subfolder="text_encoder_2"
    )

    # Load text encoders
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=cfg.revision,
        variant=cfg.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=cfg.revision,
        variant=cfg.variant,
    )

    # Determine the VAE path based on configuration
    vae_path = (
        cfg.pretrained_model_name_or_path
        if cfg.pretrained_vae_model_name_or_path is None
        else cfg.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if cfg.pretrained_vae_model_name_or_path is None else None,
        revision=cfg.revision,
        variant=cfg.variant,
    )

    # Load the UNet model
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="unet",
        revision=cfg.revision,
        variant=cfg.variant,
    )

    # Set model precision
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        cfg.mixed_precision, torch.float32
    )

    # Move models to the specified device with appropriate dtype
    for model in [unet, vae, text_encoder_one, text_encoder_two]:
        model.to(cfg.device, dtype=weight_dtype).eval()

    # Initialize ControlNet pipeline.
    controlnet = ControlNetModel.from_pretrained(cfg.pretrained_controlnet_name_or_path, torch_dtype=torch.float16)

    # Initialize the Stable Diffusion pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        controlnet=controlnet,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        unet=unet,
        torch_dtype=torch.float16,
        add_watermarker=False
    )

    # Enable FreeU if specified in the configuration
    if cfg.use_freeu:
        pipe.enable_freeu(b1=1.0, b2=1.1, s1=0.9, s2=0.2)

    # Move the pipeline to the device and set the dtype
    pipe = pipe.to(cfg.device).to(weight_dtype)

    # Configure the scheduler
    scheduler_args = {}
    variance_type = pipe.scheduler.config.get("variance_type")
    if variance_type in ["learned", "learned_range"]:
        scheduler_args["variance_type"] = "fixed_small"

    # Initialize the scheduler with updated configuration
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, **scheduler_args
    )

    # Initialize IPAdapter with the pipeline and configuration
    visual_persona = Visual_Persona(
        pipe,
        cfg.encoder_ckpt,
        cfg.decoder_ckpt,
        device=cfg.device,
        num_tokens=cfg.visual_persona_tokens,
        use_dino=cfg.use_dino,
        decoder_type=cfg.decoder_type,
    )

    return visual_persona


@pyrallis.wrap()
def main(cfg: Config):

    # Display configuration settings
    print(
        f"""====Input/Output Directory====
Output path: {cfg.out_dir}
Inputs path: {cfg.inputs_path}
Full-Body Mask path: {cfg.full_mask_path}
Body-Part Mask path: {cfg.part_mask_path}
====Configuration Settings====
Use DINO: {cfg.use_dino}
Use custom prompts: {cfg.customize_prompt}
Decoder type: {cfg.decoder_type}
====Model Checkpoints====
Diffusion checkpoint: {cfg.pretrained_model_name_or_path}
VAE checkpoint: {cfg.pretrained_vae_model_name_or_path}
Encoder checkpoint: {cfg.encoder_ckpt}
Decoder checkpoint: {cfg.decoder_ckpt}
====Inference Settings====
CUDA device: {cfg.device}
Seed: {cfg.seed}
Controlnet scale: {cfg.control_scale}
Adapter scale: {cfg.adapter_attention_scale}
Inference steps: {cfg.num_inference_steps}
Guidance scale: {cfg.guidance_scale}
Images per prompt: {cfg.num_images_per_prompt}"""
    )

    # Set precision based on the configuration
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        cfg.mixed_precision, torch.float32
    )

    # Collect input image paths
    paths = (
        [cfg.inputs_path]
        if cfg.inputs_path.is_file()
        else list(cfg.inputs_path.glob("**/*[.png,.jpg,.jpeg]"))
    )
    # Create output directory and save configuration as JSON
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()
    }

    with open(cfg.out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=4)

    # Load the mapping (ID, Text Description, Body Parts)
    # For better performance, you can provide a detailed text description of the person.
    mapping = {
        "id_0": ("man", [0, 1, 2, 3]),
    }

    # Initialize the pipeline
    vp_model = init_pipeline(cfg)

    # Image processor
    transform = CLIPImageProcessor()

    # Load Openpose model
    openpose = OpenposeDetector.from_pretrained(cfg.pretrained_openpose_name_or_path)

    # Collect control image paths
    control_paths = [cfg.control_paths] if cfg.control_paths.is_file() else list(cfg.control_paths.glob('**/*[.png,.jpg,.jpeg]'))
    
    control_imgs = {}
    for control_path in tqdm(control_paths, desc=" images", position=0):
        filename = os.path.basename(control_path)  
        file_number = os.path.splitext(filename)[0]  

        control_image_pil = Image.open(control_path).convert('RGB')        
        max_dim = max(control_image_pil.size)
        control_image_pil = ImageOps.pad(control_image_pil, (max_dim, max_dim), color=(0, 0, 0))
        control_image = transform(control_image_pil, return_tensors="pt").pixel_values.to(cfg.device).to(weight_dtype)

        numpy_control_image = control_image[0].cpu().permute(1, 2, 0).numpy()
        numpy_control_image = (numpy_control_image * 255).astype(np.uint8)
        rgb_control_image = Image.fromarray(numpy_control_image)
        openpose_control_image = openpose(rgb_control_image)

        control_imgs[file_number] = openpose_control_image

    # Set seed and generate seeds for prompts
    torch.manual_seed(cfg.seed)
    seeds = torch.randint(0, 1000000, (cfg.num_images_per_prompt,)).tolist()

    for path in tqdm(paths, desc=" Actors", position=0):

        actor_id = path.stem
        print(f"Current Actor ID: {actor_id}")

        image_out_dir = cfg.out_dir / path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        input_image_pil = Image.open(path).convert("RGB")
        input_mask = os.path.join(cfg.full_mask_path, f"{path.stem}.png")
        input_mask = Image.open(input_mask).convert("L")

        body_part_masks = {}
        body_part_keys = mapping[actor_id][1]
        for k in body_part_keys:
            body_part_mask_path = os.path.join(cfg.part_mask_path, actor_id, f"{k}.png")
            body_part_mask = Image.open(body_part_mask_path).convert("L")
            body_part_masks[k] = body_part_mask

        input_image = prepare_input_image(
            input_image_pil,
            input_mask,
            body_part_masks,
            body_part_keys,
            transform,
            cfg.device,
            weight_dtype,
            image_out_dir,
        )

        for prompt_number, prompt in enumerate(cfg.prompts):
            openpose_image = control_imgs[f"{prompt_number}"]
            openpose_image.save(image_out_dir / f"control_{prompt_number}.png")
            
            if cfg.customize_prompt:
                prompt = prompt.replace("person", mapping[actor_id][0])

            print(f"Current Prompt: {prompt}")

            generate_and_save_images_controlnet(
                vp_model,
                input_image,
                openpose_image,
                cfg.control_scale,
                cfg,
                prompt,
                seeds,
                image_out_dir,
                prompt_number,
                prompt_number,
                body_part_keys,
            )


if __name__ == "__main__":

    main()
