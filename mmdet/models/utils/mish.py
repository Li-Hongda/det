
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import ACTIVATION_LAYERS

@ACTIVATION_LAYERS.register_module()
class Mish(nn.Module):
    """Mish activation function."""
    def __init__(self,inplace : bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * (torch.tanh(F.softplus(input)))



