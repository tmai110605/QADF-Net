"""
cmig_qadf/models/backbones.py
=============================
Spatial feature map extraction (B × C × H × W)
for all 3 backbones: EfficientNet-B4, ConvNeXt-Tiny, Swin-Tiny.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b4, EfficientNet_B4_Weights,
    convnext_tiny,   ConvNeXt_Tiny_Weights,
    swin_t,          Swin_T_Weights,
)


class SpatialFundusBackbone(nn.Module):
    """
    Backbone returning spatial feature map B × C × H' × W' (prior to pooling).
    For input image 448 × 448, H' = W' = 14 across all three architectures.
    """
    FEATURE_DIMS = {
        "efficientnet_b4": 1792,
        "convnext_tiny":   768,
        "swin_tiny":       768,
    }

    def __init__(self, backbone_name: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = self.FEATURE_DIMS.get(backbone_name)
        if self.feature_dim is None:
            raise ValueError(f"Backbone '{backbone_name}' is not supported. Choose from: {list(self.FEATURE_DIMS.keys())}")

        if backbone_name == "efficientnet_b4":
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            base = efficientnet_b4(weights=weights)
            self.features = base.features
            self.arch = "efficientnet"

        elif backbone_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = convnext_tiny(weights=weights)
            self.features = base.features
            self.arch = "convnext"

        elif backbone_name == "swin_tiny":
            weights = Swin_T_Weights.DEFAULT if pretrained else None
            base = swin_t(weights=weights)
            self.features = base.features
            self.arch = "swin"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3, H, W)
        Output: (B, C, H', W') with H'=W'=14 when input=448x448
        """
        if self.arch == "efficientnet":
            # (B, 1792, H/32, W/32)
            return self.features(x)

        elif self.arch == "convnext":
            # (B, 768, H/32, W/32)
            return self.features(x)

        elif self.arch == "swin":
            # Swin features return Channels-Last (B, H/32, W/32, 768)
            feat = self.features(x)
            # Permute to PyTorch standard (B, 768, H/32, W/32)
            return feat.permute(0, 3, 1, 2).contiguous()
