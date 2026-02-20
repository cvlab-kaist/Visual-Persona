import torch

ckpt = "/media/dataset1/adobe-custimization/linear_Weight/pytorch_model_4.bin"
sd = torch.load(ckpt, map_location="cpu")
image_proj_sd = {}
ip_sd = {}
unet_sd = {}
for k in sd:
    if k.startswith("unet"):
        unet_sd[k.replace("unet.", "")] = sd[k]
        print(sd[k])
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

torch.save(
    {"image_proj": image_proj_sd, "ip_adapter": ip_sd, "unet": unet_sd},
    "/home/cvlab12/project/jisu/adobe-personalization/weights/phi_stage_2_dino_local_linear_28000.bin",
)
