import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork


class FPN(nn.Module):
    def __init__(
        self, in_channels_list, out_channels=256, extra_blocks=None, norm_layer=None
    ):
        super(FPN, self).__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels, extra_blocks, norm_layer
        )

    def forward(self, x):
        return self.fpn(x)
