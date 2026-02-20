import os
import torch
import numpy as np
import torch.nn as nn
import math
from PIL import Image
from typing import List
from safetensors import safe_open
from einops import rearrange, repeat
from model.model_utils import get_generator, is_torch2_available
from diffusers.pipelines.controlnet import MultiControlNetModel
import model.dinov2.hubconf as hubconf
from .resampler import Resampler
from .transformer import TransformerDecoder

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Pre-trained DINOv2 for Image Transformer Encoder
    """

    def __init__(self, DINOv2_ckpt, device="cuda", freeze=True):
        super().__init__()

        self.device = device

        dinov2 = hubconf.dinov2_vitg14()
        state_dict = torch.load(DINOv2_ckpt)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(dtype=torch.float16, device=self.device)

        if freeze:
            self.freeze()

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image, list):
            image = torch.cat(image, 0)

        image.to(self.device)
        features = self.model.forward_features(image)
        tokens = features["x_norm_patchtokens"]
        image_features = features["x_norm_clstoken"]
        image_features = image_features.unsqueeze(1)
        hint = torch.cat([image_features, tokens], 1)  # batch,257,1024

        return hint

    def encode(self, image):
        return self(image)


class BP_Transformer(nn.Module):
    """
    Body-partitioned Transformer Decoder
    """

    def __init__(
        self,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dim=2048,
        encoder_feat_dim=1536,
        num_queries=256,
    ):
        super().__init__()

        self.encoder_feat_dim = encoder_feat_dim

        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_queries, transformer_dim)
            * (1.0 / transformer_dim) ** 0.5
        )
        self.new_pos_embed = nn.Parameter(
            torch.randn(5, num_queries, transformer_dim)
            * (1.0 / transformer_dim) ** 0.5
        )

        self.transformer = TransformerDecoder(
            block_type="cond",
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=None,
        )

    def forward(self, image_feats):

        assert (
            image_feats.shape[-1] == self.encoder_feat_dim
        ), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        x = self.new_pos_embed
        x = self.transformer(
            x,
            cond=image_feats,
        )

        return x


class Visual_Persona:

    def __init__(
        self,
        sd_pipe,
        encoder_ckpt,
        decoder_ckpt,
        device,
        use_dino,
        num_tokens=256,
        decoder_type="Linear",
    ):
        self.device = device
        self.decoder_ckpt = decoder_ckpt
        self.num_tokens = num_tokens
        self.use_dino = use_dino
        self.decoder_type = decoder_type
        self.num_tokens = num_tokens
        self.image_encoder = FrozenDinoV2Encoder(
            encoder_ckpt, device=self.device, freeze=True
        )
        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()
        self.image_proj_model = self.init_proj().to(self.device, dtype=torch.float16)
        self.load_ip_adapter()

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)

        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(
                        CNAttnProcessor(num_tokens=self.num_tokens)
                    )
            else:
                self.pipe.controlnet.set_attn_processor(
                    CNAttnProcessor(num_tokens=self.num_tokens)
                )

    def load_ip_adapter(self):

        print("Processing: Loading Checkpoint...")
        if os.path.splitext(self.decoder_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.decoder_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = (
                            f.get_tensor(key)
                        )
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = (
                            f.get_tensor(key)
                        )
        else:
            state_dict = torch.load(self.decoder_ckpt, map_location="cpu")

        if "unet" in state_dict.keys():
            self.pipe.unet.load_state_dict(state_dict["unet"])

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    def init_proj(self):
        # Image Encoder: CLIP
        if not self.use_dino:
            if self.decoder_type == "Linear":
                image_proj_model = Resampler(
                    dim=1280,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries=self.num_tokens,
                    embedding_dim=self.image_encoder.config.hidden_size,
                    output_dim=2048,
                    ff_mult=4,
                ).to(self.device)
                print("[Current Version] Encoder: CLIP | Decoder: Linear")

            elif self.decoder_type == "Transformer":
                image_proj_model = BP_Transformer(
                    transformer_layers=8,
                    transformer_heads=8,
                    transformer_dim=2048,
                    encoder_feat_dim=1280,
                    num_queries=self.num_tokens,
                )
                print(
                    "[Current Version] Encoder: CLIP | Decoder: Body-Paritioned Transformer"
                )

            else:
                RuntimeError("No corresponding image proj model")

        # Image Encoder: DINOv2
        else:
            if self.decoder_type == "Linear":
                image_proj_model = Resampler(
                    dim=1536,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries=self.num_tokens,
                    embedding_dim=1536,
                    output_dim=2048,
                    ff_mult=4,
                ).to(self.device)
                print("[Current Version] Encoder: DINOv2 | Decoder: Linear")

            elif self.decoder_type == "Transformer":
                image_proj_model = BP_Transformer(
                    transformer_layers=8,
                    transformer_heads=8,
                    transformer_dim=2048,
                    encoder_feat_dim=1536,
                    num_queries=self.num_tokens,
                )
                print(
                    "[Current Version] Encoder: DINOv2 | Decoder: Body-Paritioned Transformer"
                )

            else:
                RuntimeError("No corresponding image proj model")

        return image_proj_model

    def get_image_embeds(self, image, use_dino, decoder_type, skip_uncond=False):

        image = image.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            if use_dino:
                image_embeds = self.image_encoder.encode(image)
            else:
                image_embeds = self.image_encoder(
                    image, output_hidden_states=True
                ).hidden_states[
                    -2
                ]  # batch, 257, 1280 (257 = CLIP token + CLS token)

            image_embeds = image_embeds[:, 1:]

        image_prompt_embeds = self.image_proj_model(image_embeds)

        if skip_uncond:
            return image_prompt_embeds

        # unconditional embeddings
        with torch.no_grad():

            if use_dino:
                uncond_image_embeds = self.image_encoder.encode(torch.zeros_like(image))
            else:
                uncond_image_embeds = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]

            uncond_image_embeds = uncond_image_embeds[:, 1:]

        uncond_image_prompt_embeds = self.image_proj_model(uncond_image_embeds)

        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        possible_keys=None,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image, self.use_dino, self.decoder_type
        )

        image_prompt_embeds = rearrange(image_prompt_embeds, "b t c -> () b t c")
        uncond_image_prompt_embeds = rearrange(
            uncond_image_prompt_embeds, "b t c -> () b t c"
        )

        selected_image_prompt_embeds = None
        selected_uncond_image_prompt_embeds = None

        for key in possible_keys:
            part_image_prompt = image_prompt_embeds[:, key, :, :].unsqueeze(1)
            part_uncond_image_prompt = uncond_image_prompt_embeds[:, key, :, :].unsqueeze(1)

            if selected_image_prompt_embeds is None:
                selected_image_prompt_embeds = part_image_prompt
                selected_uncond_image_prompt_embeds = part_uncond_image_prompt
            else:
                selected_image_prompt_embeds = torch.cat([selected_image_prompt_embeds, part_image_prompt], dim=1)
                selected_uncond_image_prompt_embeds = torch.cat([selected_uncond_image_prompt_embeds, part_uncond_image_prompt], dim=1)
                    
        image_prompt_embeds = rearrange(
            selected_image_prompt_embeds, "b l k c -> b (l k) c"
        )
        uncond_image_prompt_embeds = rearrange(
            selected_uncond_image_prompt_embeds, "b l k c -> b (l k) c"
        )

        image_prompt_embeds = image_prompt_embeds.repeat(num_samples, 1, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            num_samples, 1, 1
        )

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
