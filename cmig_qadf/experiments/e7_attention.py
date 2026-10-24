"""
cmig_qadf/experiments/e7_attention.py
=====================================
E7: Spatial attention maps across clean vs degraded View 2 inputs (Figure 4).
Extracts attention maps and attention mass allocation for report and Figure 4.
"""

import os
import json
import torch
from typing import Dict, Any

from cmig_qadf.models.dualview_model import DualViewGradingModel
from cmig_qadf.data.adapter import build_pair_records
from cmig_qadf.eval.attention_analysis import analyze_attention_e7


def run_e7_experiment(
    checkpoint_path: str,
    deepdrid_dir: str,
    test_patients: list,
    output_dir: str = "results/processed",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "e7_attention_summary.json")

    # Load model
    model = DualViewGradingModel(backbone_name="efficientnet_b4", pretrained=False)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    records = build_pair_records(deepdrid_dir, test_patients)
    res = analyze_attention_e7(model, records, device=device, output_path=out_json)
    return res
