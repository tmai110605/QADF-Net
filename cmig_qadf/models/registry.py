"""
cmig_qadf/models/registry.py
============================
Registry and Factory mapping method configurations to model instances.
"""

from typing import Dict, Any, Optional
import torch.nn as nn
from cmig_qadf.models.dualview_model import DualViewGradingModel


def build_model(
    config: Dict[str, Any],
    q_estimator: Optional[nn.Module] = None
) -> DualViewGradingModel:
    """
    Factory constructing diagnostic dual-view model from config dictionary.
    """
    model_cfg = config.get("model", {})
    method_cfg = config.get("method", {})
    quality_cfg = config.get("quality", {})

    backbone_name = method_cfg.get("backbone", model_cfg.get("backbone", "efficientnet_b4"))
    fusion_type   = method_cfg.get("fusion_type", "qc_sca")
    qc_mode       = method_cfg.get("qc_mode", "all")
    use_sink_token = method_cfg.get("use_sink_token", True)
    token_dim     = method_cfg.get("token_dim", model_cfg.get("token_dim", 128))
    grid_size     = model_cfg.get("grid_size", 14)
    num_heads     = method_cfg.get("num_heads", model_cfg.get("num_heads", 4))
    dropout       = method_cfg.get("dropout", model_cfg.get("dropout", 0.1))
    q_source      = quality_cfg.get("source", "learned")
    noisy_sigma   = quality_cfg.get("noisy_sigma", 0.1)
    num_classes   = model_cfg.get("num_classes", 5)

    model = DualViewGradingModel(
        backbone_name=backbone_name,
        pretrained=True,
        token_dim=token_dim,
        grid_size=grid_size,
        num_heads=num_heads,
        dropout=dropout,
        fusion_type=fusion_type,
        qc_mode=qc_mode,
        use_sink_token=use_sink_token,
        q_source=q_source,
        noisy_sigma=noisy_sigma,
        q_estimator=q_estimator,
        num_classes=num_classes,
    )
    return model
