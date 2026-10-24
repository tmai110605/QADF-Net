"""
cmig_qadf/experiments/e5_mechanisms.py
======================================
E5: QC-SCA mechanism ablation analysis (Table 3 Part B).
Variants: none (Standard Cross-Attn), scale_k, scale_v (Proposition 1), bias_logq (Sink Token), all.
Evaluated on clean data and under controlled degradation stress benchmark.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def get_jobs(config: dict, profile: str = "full") -> List[dict]:
    seeds = [42] if profile == "smoke" else [42, 43, 44]
    mechanisms = ["none", "scale_k", "scale_v", "bias_logq", "all"]
    backbone = config.get("model", {}).get("backbone", "efficientnet_b4")

    jobs = []
    for mech in mechanisms:
        for seed in seeds:
            job_id = f"e5_qc_sca_{backbone}_mech_{mech}_seed{seed}"
            jobs.append({
                "job_id": job_id,
                "stage": "S2",
                "experiment": "E5",
                "method_name": "qc_sca",
                "seed": seed,
                "train_mode": "train_clean",
                "method": {
                    "fusion_type": "qc_sca",
                    "qc_mode": mech,
                    "backbone": backbone,
                    "use_sink_token": (mech in ["bias_logq", "all"])
                },
                "data": config["data"],
                "model": config["model"],
                "train": config["train"],
                "quality": config["quality"],
            })
    return jobs


def aggregate_e5(raw_s3_dir: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "e5_summary.json")
    results = {}

    if not os.path.exists(raw_s3_dir):
        return results

    for fname in os.listdir(raw_s3_dir):
        if not fname.startswith("e5_") or not fname.endswith(".parquet"):
            continue
        parts = fname.replace(".parquet", "").split("_")
        # e.g. e5_qc_sca_..._mech_scale_v_seed42
        name = fname.replace(".parquet", "")
        mech = "all"
        for candidate in ["bias_logq", "scale_v", "scale_k", "none", "all"]:
            if f"_mech_{candidate}_" in name or name.endswith(f"_mech_{candidate}"):
                mech = candidate
                break

        df = pd.read_parquet(os.path.join(raw_s3_dir, fname))
        from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy
        qwk = compute_qwk(df["label"].values, df["pred"].values)
        acc = compute_accuracy(df["label"].values, df["pred"].values)

        if mech not in results:
            results[mech] = []
        results[mech].append({"qwk": qwk, "acc": acc})

    summary = {}
    for mech, runs in results.items():
        summary[mech] = {
            "qwk_mean": float(np.mean([r["qwk"] for r in runs])),
            "qwk_std": float(np.std([r["qwk"] for r in runs])),
            "acc_mean": float(np.mean([r["acc"] for r in runs])),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
