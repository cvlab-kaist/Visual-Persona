# Code adapted from https://github.com/tencent-ailab/IP-Adapter
#
# Original code is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from einops import rearrange, repeat
from safetensors import safe_open
from .resampler import PerceiverAttention, Resampler, FeedForward
from .transformer import TransformerDecoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import model.dinov2.hubconf as hubconf
import torch.nn as nn


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class ModelLRM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
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

        # attributes
        self.encoder_feat_dim = encoder_feat_dim

        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_queries, transformer_dim)
            * (1.0 / transformer_dim) ** 0.5,
            requires_grad=False,
        )
        self.new_pos_embed = nn.Parameter(
            torch.randn(5, num_queries, transformer_dim)
            * (1.0 / transformer_dim) ** 0.5,
            requires_grad=True,
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
        # image: [N, C_img, H_img, W_img]
        # image_feats : [N, (H, W), C]

        assert (
            image_feats.shape[-1] == self.encoder_feat_dim
        ), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        N = image_feats.shape[0]

        x = self.new_pos_embed
        x = self.transformer(
            x,
            cond=image_feats,
        )

        return x


class ModelLRM_Self(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
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

        # attributes
        self.encoder_feat_dim = encoder_feat_dim

        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        # self.pos_embed = nn.Parameter(torch.randn(1, num_queries, transformer_dim) * (1. / transformer_dim) ** 0.5, requires_grad = False)
        # self.new_pos_embed = nn.Parameter(torch.randn(5, num_queries, transformer_dim) * (1. / transformer_dim) ** 0.5, requires_grad = True)

        self.input_mlp = nn.Linear(1536, 2048)
        self.transformer = TransformerDecoder(
            block_type="cond_self",
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=None,
        )

    def forward(self, image_feats):
        # image: [N, C_img, H_img, W_img]
        # image_feats : [N, (H, W), C]

        assert (
            image_feats.shape[-1] == self.encoder_feat_dim
        ), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        N = image_feats.shape[0]

        # x = self.new_pos_embed
        x = self.input_mlp(image_feats)
        x = self.transformer(
            x,
            cond=x,
        )

        return x


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """

    def __init__(self, device="cuda", freeze=True):
        super().__init__()

        self.device = device

        DINOv2_weight_path = "pretrained_models/ipadapter/dinov2_vitg14_pretrain.pth"
        dinov2 = hubconf.dinov2_vitg14()
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(dtype=torch.float32, device=self.device)

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


class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class ProjPlusModel(torch.nn.Module):
    def __init__(
        self,
        cross_attention_dim=768,
        id_embeddings_dim=512,
        clip_embeddings_dim=1280,
        num_tokens=4,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(
        self,
        cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class ImageProjModel_LRM(torch.nn.Module):
    """Projection Model"""

    def __init__(
        self,
        cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter(torch.nn.Module):
    def __init__(
        self,
        unet,
        image_encoder_path,
        adapter_module,
        ip_ckpt,
        device,
        use_dino,
        num_tokens=4,
        train_local=False,
        train_local_type="linear",
        lora_scale=1.0,
    ):
        super(IPAdapter, self).__init__()

        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.cross_attention_dim = 2048
        self.unet = unet
        self.dtype = torch.float32
        self.lora_scale = lora_scale
        self.adapter_modules = adapter_module
        self.use_dino = use_dino
        self.train_local = train_local

        if self.train_local:
            self.train_local_type = train_local_type
            self.num_tokens = 256
        else:
            self.train_local_type = ""

        # load image encoder
        if not use_dino:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.image_encoder_path
            ).to(self.device, dtype=self.dtype)
            self.image_encoder.requires_grad_(False)

        else:
            self.image_encoder = FrozenDinoV2Encoder(device=self.device, freeze=True)

        self.clip_image_processor = CLIPImageProcessor()

        self.image_proj_model = self.init_proj()

        if self.ip_ckpt is not None:
            self.load_from_checkpoint(ip_ckpt)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device)
        return image_proj_model

    def load_from_checkpoint(self, ckpt_path: str):

        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(
            torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()])
        )
        orig_adapter_sum = torch.sum(
            torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()])
        )

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)

        pretrained_pos_embed = state_dict["image_proj"]["pos_embed"]
        self.image_proj_model.new_pos_embed.data.copy_(
            pretrained_pos_embed.repeat(5, 1, 1)
        )
        self.image_proj_model.new_pos_embed.requires_grad = True

        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(
            torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()])
        )
        new_adapter_sum = torch.sum(
            torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()])
        )

        # Verify if the weights have changed
        assert (
            orig_ip_proj_sum != new_ip_proj_sum
        ), "Weights of image_proj_model did not change!"
        assert (
            orig_adapter_sum != new_adapter_sum
        ), "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            pil_image = (pil_image + 1) // 2
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=torch.float32)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float32)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(clip_image_embeds)
        )
        return image_prompt_embeds, uncond_image_prompt_embeds

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        unet_added_cond_kwargs,
        pil_image,
        clip_image_embeds=None,
    ):
        ip_tokens, _ = self.get_image_embeds(pil_image, clip_image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
        ).sample

        return noise_pred


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        if not self.use_dino:
            if not self.train_local:
                image_proj_model = Resampler(
                    dim=1280,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries=self.num_tokens,
                    embedding_dim=self.image_encoder.config.hidden_size,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4,
                ).to(self.device)
                print("Not Local!")

            elif self.train_local and self.train_local_type == "linear":
                image_proj_model = Resampler(
                    dim=1280,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries=self.num_tokens,
                    embedding_dim=self.image_encoder.config.hidden_size,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4,
                ).to(self.device)
                print("Linear!")

            elif self.train_local and self.train_local_type == "LRM":
                image_proj_model = ModelLRM(
                    transformer_layers=8,
                    transformer_heads=8,
                    transformer_dim=self.cross_attention_dim,
                    encoder_feat_dim=1280,
                    num_queries=self.num_tokens,
                )
                print("LRM!")
                # default settings
                # transformer_layers: 12
                # transformer_heads: 8

            else:
                RuntimeError("No corresponding image proj model")

        else:
            if not self.train_local:
                image_proj_model = Resampler(
                    dim=1536,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries=self.num_tokens,
                    embedding_dim=1536,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4,
                ).to(self.device)
                print("Not Local!")

            elif self.train_local and self.train_local_type == "linear":
                image_proj_model = Resampler(
                    dim=1536,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries=self.num_tokens,
                    embedding_dim=1536,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4,
                ).to(self.device)
                print("Linear!")

            elif self.train_local and self.train_local_type == "LRM":
                image_proj_model = ModelLRM(
                    transformer_layers=8,
                    transformer_heads=8,
                    transformer_dim=self.cross_attention_dim,
                    encoder_feat_dim=1536,
                    num_queries=self.num_tokens,
                )
                print("LRM!")
                # default settings
                # transformer_layers: 12
                # transformer_heads: 8

            else:
                RuntimeError("No corresponding image proj model")

        return image_proj_model

    def get_image_embeds(
        self, clip_image, use_dino, train_local, train_local_type, skip_uncond=False
    ):

        clip_image = clip_image.to(self.device, dtype=self.dtype)

        # clip image embeddings
        with torch.no_grad():
            if use_dino:
                clip_image_embeds = self.image_encoder.encode(clip_image)
            else:
                clip_image_embeds = self.image_encoder(
                    clip_image, output_hidden_states=True
                ).hidden_states[
                    -2
                ]  # batch, 257, 1280 (257 = CLIP token + CLS token)

            if train_local:
                clip_image_embeds = clip_image_embeds[:, 1:]

        image_prompt_embeds = self.image_proj_model(
            clip_image_embeds
        )  # output = batch, 16, 2048

        if skip_uncond:
            return image_prompt_embeds

        # clip image embeddings for unconditioned (drop-out)
        with torch.no_grad():

            if use_dino:
                uncond_clip_image_embeds = self.image_encoder.encode(
                    torch.zeros_like(clip_image)
                )
            else:
                uncond_clip_image_embeds = self.image_encoder(
                    torch.zeros_like(clip_image), output_hidden_states=True
                ).hidden_states[-2]

            if train_local:
                uncond_clip_image_embeds = uncond_clip_image_embeds[:, 1:]

        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)

        return image_prompt_embeds, uncond_image_prompt_embeds

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        unet_added_cond_kwargs,
        clip_image,
        train_with_cropped,
        train_with_pos,
        editing,
        drop_rate=0,
        skip_uncond=False,
    ):
        adapter_states_cond, adapter_states_uncond = self.get_image_embeds(
            clip_image,
            self.use_dino,
            self.train_local,
            self.train_local_type,
            skip_uncond,
        )

        batch_size = unet_added_cond_kwargs["batch_size"]

        if train_with_cropped:
            adapter_states_cond = adapter_states_cond.chunk(batch_size, dim=0)
            adapter_states_cond = torch.cat(
                [
                    rearrange(c_states, "b1 t c -> () (b1 t) c")
                    for c_states in adapter_states_cond
                ],
                dim=0,
            )
            adapter_states_uncond = adapter_states_uncond.chunk(batch_size, dim=0)
            adapter_states_uncond = torch.cat(
                [
                    rearrange(u_states, "b1 t c -> () (b1 t) c")
                    for u_states in adapter_states_uncond
                ],
                dim=0,
            )

            # if train_with_pos:
            #     if editing:
            #         pos = positionalencoding1d(adapter_states_cond.shape[-1], 4)[None,...].to(self.device, dtype=self.dtype)
            #     else:
            #         pos = positionalencoding1d(adapter_states_cond.shape[-1], 5)[None,...].to(self.device, dtype=self.dtype)
            #     pos = repeat(pos, 'b l c -> b (l k) c', k = 256)
            #     adapter_states_cond += pos
            #     adapter_states_uncond += pos

        if drop_rate > 0:
            idx_to_replace = torch.rand(len(adapter_states_cond)) < drop_rate
            print("idx_to_replace", idx_to_replace)
            adapter_states_cond[idx_to_replace] = adapter_states_uncond[idx_to_replace]

        ip_tokens = adapter_states_cond

        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
        ).sample
        return noise_pred
