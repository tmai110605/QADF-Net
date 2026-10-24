"""
cmig_qadf/experiments/e2_degradation.py
=======================================
E2: Controlled degradation benchmark (Core experimental benchmark of the CMIG paper).
7 degradation types × 5 severity levels × {View 1, View 2, both}.
Two training regimes: train_clean and train_aug.
Produces data for Figure 3 (QWK curves) and Tables S1, S2 (AUDC, Harm Rate, PFR).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from cmig_qadf.data.degradations import DEGRADATION_TYPES


def get_jobs(config: dict, profile: str = "full") -> List[dict]:
    """Generate training and evaluation jobs for E2 experiment."""
    seeds = config.get("seeds", [42, 43, 44, 45, 46])
    methods = config.get("methods", ["qc_sca", "concat", "average", "q_weighted", "cross_attn"])
    backbone = config.get("model", {}).get("backbone", "efficientnet_b4")

    jobs = []
    # Training regimes: train_clean (inherited from E1) and train_aug (augmented training)
    for method in methods:
        for seed in seeds:
            job_id = f"e2_{method}_{backbone}_seed{seed}_aug"
            jobs.append({
                "job_id": job_id,
                "stage": "S2",
                "experiment": "E2",
                "method_name": method,
                "seed": seed,
                "train_mode": "train_aug",
                "method": {"fusion_type": method, "backbone": backbone},
                "data": config["data"],
                "model": config["model"],
                "train": {**config["train"], "train_mode": "train_aug"},
                "quality": config["quality"],
            })
    return jobs


def aggregate_e2(raw_s4_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Aggregate E2 degradation results from results/raw/S4/.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "e2_summary.json")

    results = {
        "train_clean": {},
        "train_aug": {},
    }

    if not os.path.exists(raw_s4_dir):
        return results

    for fname in os.listdir(raw_s4_dir):
        if not fname.endswith("_degradations.json"):
            continue
        filepath = os.path.join(raw_s4_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse job name
        name = fname.replace("_degradations.json", "")
        # e.g. e1_qc_sca_... or e2_qc_sca_...
        mode = "train_aug" if "_aug" in name else "train_clean"
        sub = name[3:] if (name.startswith("e1_") or name.startswith("e2_")) else name
        if sub.startswith("q_weighted"):
            continue  # Exclude Q-Weighted per benchmark guidelines
        method = "other"
        for m in [
            "qc_sca", "best_single_oracle", "best_single_learned",
            "cross_attn", "co_attn", "single_v1", "single_v2",
            "concat", "average", "max"
        ]:
            if sub.startswith(m):
                method = m
                break

        if method not in results[mode]:
            results[mode][method] = []
        results[mode][method].append(data)

    # Compute mean AUDC across degradation types
    aggregated = {}
    for mode, methods_dict in results.items():
        aggregated[mode] = {}
        for m, run_list in methods_dict.items():
            aggregated[mode][m] = {
                "mean_clean_qwk": float(np.mean([r.get("clean_qwk", 0.0) for r in run_list])) if run_list else 0.0,
                "audc_per_type": {},
            }
            if not run_list:
                continue

            first_run = run_list[0]
            for v in first_run.get("curves", {}):
                aggregated[mode][m]["audc_per_type"][v] = {}
                for d_type in first_run["curves"][v]:
                    audcs = [r["curves"][v][d_type]["audc"] for r in run_list if v in r.get("curves", {}) and d_type in r["curves"][v]]
                    aggregated[mode][m]["audc_per_type"][v][d_type] = float(np.mean(audcs)) if audcs else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated
