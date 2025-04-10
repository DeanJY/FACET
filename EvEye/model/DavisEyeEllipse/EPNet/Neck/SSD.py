import torch
import torch.nn as nn
import timm
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import SSD, SSDHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.backbone_utils import (
    BackboneWithFPN,
    resnet_fpn_backbone,
)
