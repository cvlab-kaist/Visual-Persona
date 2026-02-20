import torch
import torch.nn.functional as F

class Loss:
    """
    General purpose loss class.
    """

    def __init__(self, dtype=torch.float32, accelerator=None, **kwargs):
        self.iteration = -1
        self.dtype = dtype
        self.accelerator = accelerator

    def __call__(self, **kwargs):
        self.iteration += 1
        return self.forward(**kwargs)


class NoiseLoss(Loss):
    """
    Regular diffusion loss between predicted noise and target noise.

    Args:
        predicted_noise (torch.Tensor): noise predicted by the diffusion model
        target_noise (torch.Tensor): actual noise added to the image.
    """

    def forward(
        self, predicted_noise: torch.Tensor, target_noise: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return F.mse_loss(
            predicted_noise.float(), target_noise.float(), reduction="mean"
        )
