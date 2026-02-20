import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

attn_maps = {}


def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split(".")[-1].startswith("attn2"):
            print(hook_fn(name))
            module.register_forward_hook(hook_fn(name))

    return unet


def upscale(attn_map, target_size):

    attn_map = torch.mean(attn_map, dim=0)  # torch.Size([10, 4096, 1024])
    attn_map = attn_map.permute(1, 0)  # torch.Size([1024, 4096])
    # temp_size = None
    spatial_size = int(torch.sqrt(torch.tensor([attn_map.shape[-1]])))

    # for i in range(0,5):
    #     scale = 2 ** i
    #     if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
    #         temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
    #         break

    # assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(
        attn_map.shape[0], spatial_size, spatial_size
    )  # torch.Size([1024, 64, 64])

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=64,
        mode="bilinear",
        align_corners=False,
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    # torch.Size([1024, 1024(H), 1024(W)])
    return attn_map


# def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

#     idx = 0 if instance_or_negative else 1
#     net_attn_maps = []

#     for name, attn_map in attn_maps.items():
#         attn_map = attn_map.cpu() if detach else attn_map # torch.Size([2, 10, 4096, 1024])
#         attn_map = torch.chunk(attn_map, 2)[idx].squeeze() # torch.Size([2, 10, 4096, 1024])

#         attn_map = upscale(attn_map, image_size)

#         net_attn_maps.append(attn_map)

#     net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

#     return net_attn_maps

# Initialize a variable to keep track of the cumulative sum and count of attention maps


def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
    cumulative_sum = None
    count = 0
    # idx = 0 if instance_or_negative else 1
    idx = 0

    for name, attn_map in attn_maps.items():
        # Move the attention map to the CPU if necessary and chunk it
        attn_map = (
            attn_map.cpu() if detach else attn_map
        )  # torch.Size([2, 10, 4096, 1024])
        attn_map = torch.chunk(attn_map, 2)[
            idx
        ].squeeze()  # torch.Size([10, 4096, 1024])

        # Upscale the attention map to the desired image size
        attn_map = upscale(attn_map, image_size)

        # Accumulate the sum of the attention maps incrementally
        if cumulative_sum is None:
            cumulative_sum = attn_map
        else:
            cumulative_sum += attn_map

        count += 1  # Increment the count of processed attention maps

    # Compute the mean of the attention maps
    net_attn_maps = cumulative_sum / count

    return net_attn_maps


def attnmaps2images(net_attn_maps):

    # total_attn_scores = 0
    total_attention = None
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        if total_attention is None:
            total_attention = attn_map
        else:
            total_attention += attn_map
        # total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (
            (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        )
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        # print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        # image = fix_save_attn_map(attn_map)
        images.append(image)

    total_attention = total_attention / len(net_attn_maps)
    normalized_attn_map = (
        (total_attention - np.min(total_attention))
        / (np.max(total_attention) - np.min(total_attention))
        * 255
    )
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    image = Image.fromarray(normalized_attn_map)

    # print(total_attn_scores)
    return images, [image]


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [
                torch.Generator(device).manual_seed(seed_item) for seed_item in seed
            ]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator
