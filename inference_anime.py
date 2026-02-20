from tqdm import tqdm
import json
import pyrallis
import torch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
from PIL import Image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
from transformers import CLIPImageProcessor
from model.visual_persona.vp_test import Visual_Persona
from model.model_utils import import_model_class_from_model_name_or_path

from inference_utils.utils import (
    prepare_input_image,
    generate_and_save_images,
)


@dataclass
class Config:
    # Path to the input images
    inputs_path: Path = Path("data_inference/evaluation_anime/images")
    full_mask_path: Path = Path("data_inference/evaluation_anime/full_body_masks")
    part_mask_path: Path = Path("data_inference/evaluation_anime/body_part_masks")
    out_dir: Path = Path("./results/anime")

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

    # Prompt settings
    with open("data_inference/evaluation_anime/prompts.txt", "r") as f:
        prompts = f.read().split("\n")

    # Inference settings
    device: str = "cuda:2"
    seed: int = 77
    adapter_attention_scale: float = 0.7
    num_inference_steps: int = 50
    guidance_scale: float = 12
    num_images_per_prompt: int = 4
    img_size: int = 1024
    visual_persona_tokens: int = 256
    mixed_precision: str = "fp16"
    use_freeu: bool = False
    revision: Optional[str] = None
    variant: Optional[str] = None
    negative_prompt: str = (
        "bag, low quality, bad quality, NSFW, ugly, disfigured, deformed, three arms, three legs, fused fingers, too many finger, photo, deformed, black and white, realism, disfigured, low contrast"
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

    # Initialize the Stable Diffusion pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        unet=unet,
        torch_dtype=torch.float16,
        add_watermarker=False,
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
        "1": ("man", [1, 2, 3, 4]),
        "2": ("man", [1, 2, 3, 4]),
        "3": ("man", [1, 2, 3, 4]),
        "4": ("man", [1, 2, 3, 4]),
        "5": ("man", [1, 2, 3, 4]),
        "6": ("man", [1, 2, 3, 4])
    }

    # Initialize the pipeline
    vp_model = init_pipeline(cfg)

    # Image processor
    transform = CLIPImageProcessor()

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

            if cfg.customize_prompt:
                prompt = prompt.replace("person", mapping[actor_id][0])

            print(f"Current Prompt: {prompt}")

            generate_and_save_images(
                vp_model,
                input_image,
                cfg,
                prompt,
                seeds,
                image_out_dir,
                prompt_number,
                body_part_keys,
            )


if __name__ == "__main__":

    main()
