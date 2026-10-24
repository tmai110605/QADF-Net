"""
cmig_qadf/experiments/e9_efficiency.py
======================================
E9: Computational efficiency profiling (Table 4 Part B).
Measures FLOPs, parameter counts, inference latency (ms), and peak VRAM on standardized hardware.
"""

import os
import json
import torch
from typing import Dict, Any

from cmig_qadf.models.dualview_model import DualViewGradingModel
from cmig_qadf.eval.efficiency import measure_latency_and_vram


def run_e9_experiment(
    methods: list = ["qc_sca", "concat", "average", "cross_attn"],
    backbone: str = "efficientnet_b4",
    output_dir: str = "results/processed",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "e9_efficiency_summary.json")

    results = {}
    for m in methods:
        model = DualViewGradingModel(backbone_name=backbone, fusion_type=m, pretrained=False)
        bench = measure_latency_and_vram(model, input_size=448, num_warmup=5, num_runs=30, device=device)
        results[m] = bench

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
