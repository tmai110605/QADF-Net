"""
cmig_qadf/experiments/e4_qsource.py
===================================
E4: Investigation of quality source q (Table 3 Part A).
Trained with: {oracle, learned, const1}.
Evaluated with: {oracle, learned, const1, noisy sigma in {0.05, 0.1, 0.2, 0.3}}.
Fully eliminates concern of "human label leakage at inference".
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def get_jobs(config: dict, profile: str = "full") -> List[dict]:
    """Generate jobs for E4."""
    seeds = [42] if profile == "smoke" else [42, 43, 44]
    sources = ["learned", "oracle", "const1"]
    backbone = config.get("model", {}).get("backbone", "efficientnet_b4")

    jobs = []
    for src in sources:
        for seed in seeds:
            job_id = f"e4_qc_sca_{backbone}_src_{src}_seed{seed}"
            jobs.append({
                "job_id": job_id,
                "stage": "S2",
                "experiment": "E4",
                "method_name": "qc_sca",
                "seed": seed,
                "train_mode": "train_clean",
                "method": {"fusion_type": "qc_sca", "backbone": backbone},
                "data": config["data"],
                "model": config["model"],
                "train": config["train"],
                "quality": {**config["quality"], "source": src},
            })
    return jobs


def aggregate_e4(raw_s3_dir: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "e4_summary.json")
    results = {}

    if not os.path.exists(raw_s3_dir):
        return results

    for fname in os.listdir(raw_s3_dir):
        if not fname.startswith("e4_") or not fname.endswith(".parquet"):
            continue
        parts = fname.replace(".parquet", "").split("_")
        # e.g. e4_qc_sca_..._src_learned_seed42
        src = "learned"
        for i, p in enumerate(parts):
            if p == "src" and i + 1 < len(parts):
                src = parts[i+1]
                break

        df = pd.read_parquet(os.path.join(raw_s3_dir, fname))
        from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy
        qwk = compute_qwk(df["label"].values, df["pred"].values)
        acc = compute_accuracy(df["label"].values, df["pred"].values)

        if src not in results:
            results[src] = []
        results[src].append({"qwk": qwk, "acc": acc})

    summary = {}
    for src, runs in results.items():
        summary[src] = {
            "qwk_mean": float(np.mean([r["qwk"] for r in runs])),
            "qwk_std": float(np.std([r["qwk"] for r in runs])),
            "acc_mean": float(np.mean([r["acc"] for r in runs])),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
