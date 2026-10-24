"""
cmig_qadf/models/q_estimator.py
===============================
Image quality estimator q_hat (QEstimator).
Compact network (MobileNetV3-Small), input 224×224, sigmoid output ∈ [0, 1].
Trained in Stage S1 and FROZEN completely during diagnostic model training.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class QEstimator(nn.Module):
    """
    Estimates optical image quality for an individual fundus photograph.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        
        # Retain feature extractor
        self.features = base.features
        self.avgpool = base.avgpool
        
        # Replace classifier with scalar output and Sigmoid activation
        in_features = base.classifier[0].in_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3, 224, 224)
        Output: (B, 1) ∈ [0, 1]
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        q = self.head(feat)
        return q
