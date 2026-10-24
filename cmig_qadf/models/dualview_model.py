"""
cmig_qadf/models/dualview_model.py
==================================
Assembled end-to-end model (Backbone + Spatial Tokens + Fusion Module + CORAL Head).
Supports interchangeable quality sources: learned (default), oracle, const1, noisy.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from cmig_qadf.models.backbones import SpatialFundusBackbone
from cmig_qadf.models.tokens import SpatialTokenProjector
from cmig_qadf.models.heads import CoralOrdinalHead
from cmig_qadf.models.qc_sca import QualityConditionedSpatialCrossAttention
from cmig_qadf.models.fusion_baselines import (
    ConcatFusion, AverageFusion, MaxFusion,
    QWeightedAverageFusion,
    CoAttentionBilinearFusion, SingleViewSelectorFusion
)


class DualViewGradingModel(nn.Module):
    """
    Overall architecture for dual-view diabetic retinopathy grading.
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        token_dim: int = 128,
        grid_size: int = 14,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = "qc_sca",     # "qc_sca", "concat", "average", "max", "q_weighted", "cross_attn", "co_attn", "single_v1", "single_v2", "best_single_oracle", "best_single_learned"
        qc_mode: str = "all",            # For QC-SCA: "none", "scale_k", "scale_v", "bias_logq", "all"
        use_sink_token: bool = True,
        q_source: str = "learned",       # "learned", "oracle", "const1", "noisy"
        noisy_sigma: float = 0.1,
        q_estimator: Optional[nn.Module] = None,
        num_classes: int = 5,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.q_source = q_source
        self.noisy_sigma = noisy_sigma
        self.q_estimator = q_estimator

        # 1. Shared Backbone
        self.backbone = SpatialFundusBackbone(backbone_name=backbone_name, pretrained=pretrained)
        in_channels = self.backbone.feature_dim

        # 2. Token Projector (2D Positional + View Embeddings)
        self.tokens = SpatialTokenProjector(
            in_channels=in_channels,
            token_dim=token_dim,
            grid_size=grid_size
        )

        # 3. Fusion Module
        if fusion_type == "qc_sca":
            self.fusion = QualityConditionedSpatialCrossAttention(
                token_dim=token_dim, num_heads=num_heads, dropout=dropout,
                qc_mode=qc_mode, use_sink_token=use_sink_token
            )
        elif fusion_type == "cross_attn":
            # Standard cross-attention without quality conditioning
            self.fusion = QualityConditionedSpatialCrossAttention(
                token_dim=token_dim, num_heads=num_heads, dropout=dropout,
                qc_mode="none", use_sink_token=False
            )
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(token_dim=token_dim)
        elif fusion_type == "average":
            self.fusion = AverageFusion(token_dim=token_dim)
        elif fusion_type == "max":
            self.fusion = MaxFusion(token_dim=token_dim)
        elif fusion_type == "q_weighted":
            self.fusion = QWeightedAverageFusion(token_dim=token_dim)
        elif fusion_type == "co_attn":
            self.fusion = CoAttentionBilinearFusion(token_dim=token_dim, num_heads=num_heads)
        elif fusion_type == "single_v1":
            self.fusion = SingleViewSelectorFusion(mode="v1")
        elif fusion_type == "single_v2":
            self.fusion = SingleViewSelectorFusion(mode="v2")
        elif fusion_type in ["best_single_oracle", "best_single_learned"]:
            self.fusion = SingleViewSelectorFusion(mode="best_q")
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # 4. Fixed CORAL Head (shared across all methods)
        self.head = CoralOrdinalHead(in_features=token_dim, num_classes=num_classes)

    def _resolve_quality(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        q1_oracle: torch.Tensor,
        q2_oracle: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve quality score inputs according to method configuration."""
        B = img1.shape[0]

        if self.q_source == "oracle":
            q1, q2 = q1_oracle, q2_oracle

        elif self.q_source == "const1":
            q1 = torch.ones_like(q1_oracle)
            q2 = torch.ones_like(q2_oracle)

        elif self.q_source == "learned":
            if self.q_estimator is not None:
                # Downsample to 224x224 for q_estimator
                with torch.no_grad():
                    im1_small = nn.functional.interpolate(img1, size=(224, 224), mode="bilinear", align_corners=False)
                    im2_small = nn.functional.interpolate(img2, size=(224, 224), mode="bilinear", align_corners=False)
                    q1 = self.q_estimator(im1_small)
                    q2 = self.q_estimator(im2_small)
            else:
                # Safe fallback if q_estimator checkpoint is not loaded
                q1, q2 = q1_oracle, q2_oracle

        elif self.q_source == "noisy":
            # Add Gaussian noise N(0, sigma^2)
            noise1 = torch.randn_like(q1_oracle) * self.noisy_sigma
            noise2 = torch.randn_like(q2_oracle) * self.noisy_sigma
            q1 = torch.clamp(q1_oracle + noise1, 0.001, 1.0)
            q2 = torch.clamp(q2_oracle + noise2, 0.001, 1.0)
        else:
            q1, q2 = q1_oracle, q2_oracle

        return q1, q2

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        q1_raw: torch.Tensor,
        q2_raw: torch.Tensor,
    ) -> Dict[str, Any]:
        # 1. Resolve quality source
        q1, q2 = self._resolve_quality(img1, img2, q1_raw, q2_raw)

        # 2. Shared Backbone extracts spatial feature maps
        feat1 = self.backbone(img1)  # (B, C, H', W')
        feat2 = self.backbone(img2)  # (B, C, H', W')

        # 3. Project to spatial tokens (14x14)
        tokens1 = self.tokens(feat1, view_idx=1)  # (B, M, d)
        tokens2 = self.tokens(feat2, view_idx=2)  # (B, M, d)

        # 4. Feature fusion
        fusion_out = self.fusion(tokens1, tokens2, q1, q2)
        f_cons = fusion_out["f_cons"]  # (B, d)

        # 5. Fixed CORAL Head classification
        head_out = self.head(f_cons)

        return {
            **head_out,
            "f_cons": f_cons,
            "q1": q1,
            "q2": q2,
            **{k: v for k, v in fusion_out.items() if k != "f_cons"}
        }
