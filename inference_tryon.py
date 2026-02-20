import json
import pyrallis
import torch
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
    prepare_input_image_tryon,
    generate_and_save_images,
)

@dataclass
class Config:
    # Path to the input images
    input_head_path: Path = Path("data_inference/evaluation_tryon/images/image_1.png")
    input_top_path: Path = Path("data_inference/evaluation_tryon/images/image_2.png")
    input_bottom_path: Path = Path("data_inference/evaluation_tryon/images/image_3.png")
    input_shoes_path: Path = Path("data_inference/evaluation_tryon/images/image_4.png")
    part_mask_head_path: Path = Path("data_inference/evaluation_tryon/body_part_masks/image_1/1.png")
    part_mask_top_path: Path = Path("data_inference/evaluation_tryon/body_part_masks/image_2/2.png")
    part_mask_bottom_path: Path = Path("data_inference/evaluation_tryon/body_part_masks/image_3/3.png")
    part_mask_shoes_path: Path = Path("data_inference/evaluation_tryon/body_part_masks/image_4/4.png")

    out_dir: Path = Path("./results/try_on")

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
    with open("data_inference/evaluation_tryon/prompts.txt", "r") as f:
        prompts = f.read().split("\n")

    # Inference settings
    device: str = "cuda:0"
    seed: int = 77
    adapter_attention_scale: float = 0.7
    num_inference_steps: int = 50
    guidance_scale: float = 12
    num_images_per_prompt: int = 1
    img_size: int = 1024
    visual_persona_tokens: int = 256
    mixed_precision: str = "fp16"
    use_freeu: bool = False
    revision: Optional[str] = None
    variant: Optional[str] = None
    negative_prompt: str = (
        "low quality, bad quality, NSFW, ugly, disfigured, deformed, three arms, three legs, fused fingers, too many fingers, multiple people, cloned face, bad proportions, lowres, bad anatomy, worst quality"
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

    # Create output directory and save configuration as JSON
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()
    }

    with open(cfg.out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=4)

    # Initialize the pipeline
    vp_model = init_pipeline(cfg)

    # Image processor
    transform = CLIPImageProcessor()

    # Set seed and generate seeds for prompts
    torch.manual_seed(cfg.seed)
    seeds = torch.randint(0, 1000000, (cfg.num_images_per_prompt,)).tolist()

    head_path = cfg.input_head_path
    actor_id = head_path.stem
    print(f"Current Actor ID: {actor_id}")
    
    image_out_dir = cfg.out_dir / head_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    head_image_pil = Image.open(head_path).convert("RGB")
    head_mask = Image.open(cfg.part_mask_head_path).convert("L")
    top_image_pil = Image.open(cfg.input_top_path).convert("RGB")
    top_mask = Image.open(cfg.part_mask_top_path).convert("L")
    bottom_image_pil = Image.open(cfg.input_bottom_path).convert("RGB")
    bottom_mask = Image.open(cfg.part_mask_bottom_path).convert("L")
    shoes_image_pil = Image.open(cfg.input_shoes_path).convert("RGB")
    shoes_mask = Image.open(cfg.part_mask_shoes_path).convert("L")
  
    input_image = prepare_input_image_tryon(
        head_image_pil, 
        head_mask, 
        top_image_pil, 
        top_mask,
        bottom_image_pil, 
        bottom_mask,
        shoes_image_pil,
        shoes_mask,
        transform,
        cfg.device,
        weight_dtype,
        image_out_dir,
    )

    for prompt_number, prompt in enumerate(cfg.prompts):

        if cfg.customize_prompt:
            prompt = prompt.replace("person", "woman")

        print(f"Current Prompt: {prompt}")

        generate_and_save_images(
            vp_model,
            input_image,
            cfg,
            prompt,
            seeds,
            image_out_dir,
            prompt_number,
            [1,2,3,4],
        )


if __name__ == "__main__":

    main()
