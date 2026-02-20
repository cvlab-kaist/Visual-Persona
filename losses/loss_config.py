from dataclasses import dataclass, field
from typing import List


@dataclass
class SingleLossConfig:
    name: str
    weight: float = 1.0
    init_params: dict = field(default_factory=dict)


@dataclass
class LossesConfig:
    diffusion_losses: List[SingleLossConfig]
