"""
cmig_qadf/models/tokens.py
==========================
Spatial token representation:
B×C×H×W -> 1×1 Conv to d -> AdaptiveAvgPool to G×G -> Add 2D Positional Embedding + View Embedding.
Flexible configuration: G ∈ {1, 7, 14, 28}, d (default 128).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTokenProjector(nn.Module):
    """
    Transforms spatial feature maps into a spatial token sequence augmented with positional and view embeddings.
    """
    def __init__(
        self,
        in_channels: int = 1792,
        token_dim: int = 128,
        grid_size: int = 14,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        # 1x1 Conv spatial dimension reduction
        self.proj = nn.Conv2d(in_channels, token_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(token_dim)

        # Learned 2D positional embeddings: (1, G*G, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, token_dim))

        # Learned view embeddings for View 1 and View 2: (1, 1, d)
        self.view1_embed = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.view2_embed = nn.Parameter(torch.zeros(1, 1, token_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.view1_embed, std=0.02)
        nn.init.trunc_normal_(self.view2_embed, std=0.02)

    def forward(self, feat: torch.Tensor, view_idx: int = 1) -> torch.Tensor:
        """
        Input:  (B, C, H, W)
        Output: (B, G*G, token_dim)
        """
        B = feat.shape[0]

        # 1. Project via 1x1 conv to d
        x = self.proj(feat)  # (B, d, H, W)

        # 2. Adaptive average pooling to G x G grid
        if x.shape[-2:] != (self.grid_size, self.grid_size):
            x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))

        # 3. Reshape into sequence (B, G*G, d)
        x = x.flatten(2).transpose(1, 2)  # (B, G*G, d)
        x = self.norm(x)

        # 4. Add 2D positional embedding and view embedding
        v_embed = self.view1_embed if view_idx == 1 else self.view2_embed
        x = x + self.pos_embed + v_embed

        return x
