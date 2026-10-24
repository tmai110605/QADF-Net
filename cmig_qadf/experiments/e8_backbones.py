"""
cmig_qadf/experiments/e8_backbones.py
=====================================
E8: Backbone generalization study across 3 architectures (Table 4 Part A).
3 Backbones: EfficientNet-B4, ConvNeXt-Tiny, Swin-Tiny.
Methods: {best-single, concat, q-weighted, cross_attn, qc_sca}.
5 seeds (42-46) reporting mean and standard deviation.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def get_jobs(config: dict, profile: str = "full") -> List[dict]:
    seeds = [42] if profile == "smoke" else ([42, 43, 44] if profile == "fast" else [42, 43, 44, 45, 46])
    backbones = ["efficientnet_b4"] if profile in ["smoke", "fast"] else ["efficientnet_b4", "convnext_tiny", "swin_tiny"]
    methods = ["qc_sca", "concat", "q_weighted", "cross_attn", "single_v1"]

    jobs = []
    for bb in backbones:
        # Skip efficientnet_b4 if already executed in E1
        if bb == "efficientnet_b4":
            continue
        for m in methods:
            for s in seeds:
                job_id = f"e8_{m}_{bb}_seed{s}"
                jobs.append({
                    "job_id": job_id,
                    "stage": "S2",
                    "experiment": "E8",
                    "method_name": m,
                    "seed": s,
                    "train_mode": "train_clean",
                    "method": {"fusion_type": m, "backbone": bb},
                    "data": config["data"],
                    "model": {**config["model"], "backbone": bb},
                    "train": config["train"],
                    "quality": config["quality"],
                })
    return jobs


def aggregate_e8(raw_s3_dir: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "e8_summary.json")
    results = {}

    if not os.path.exists(raw_s3_dir):
        return results

    for fname in os.listdir(raw_s3_dir):
        if not (fname.startswith("e8_") or fname.startswith("e1_")) or not fname.endswith(".parquet"):
            continue
        parts = fname.replace(".parquet", "").split("_")
        name_no_ext = fname.replace(".parquet", "")
        # e.g. e8_qc_sca_convnext_tiny_seed42
        sub = name_no_ext[3:] if (name_no_ext.startswith("e8_") or name_no_ext.startswith("e1_")) else name_no_ext
        if sub.startswith("q_weighted"):
            continue  # Exclude Q-Weighted per benchmark protocol
        m = "other"
        for candidate in [
            "qc_sca", "best_single_oracle", "best_single_learned",
            "cross_attn", "co_attn", "single_v1", "single_v2",
            "concat", "average", "max"
        ]:
            if sub.startswith(candidate):
                m = candidate
                break
        bb = "efficientnet_b4"
        if "convnext_tiny" in fname: bb = "convnext_tiny"
        elif "swin_tiny" in fname: bb = "swin_tiny"

        df = pd.read_parquet(os.path.join(raw_s3_dir, fname))
        from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy
        qwk = compute_qwk(df["label"].values, df["pred"].values)
        acc = compute_accuracy(df["label"].values, df["pred"].values)

        if bb not in results: results[bb] = {}
        if m not in results[bb]: results[bb][m] = []
        results[bb][m].append({"qwk": qwk, "acc": acc})

    summary = {}
    for bb, m_dict in results.items():
        summary[bb] = {}
        for m, runs in m_dict.items():
            summary[bb][m] = {
                "qwk_mean": float(np.mean([r["qwk"] for r in runs])),
                "qwk_std": float(np.std([r["qwk"] for r in runs])),
                "acc_mean": float(np.mean([r["acc"] for r in runs])),
            }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
