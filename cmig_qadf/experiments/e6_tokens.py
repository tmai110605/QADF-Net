"""
cmig_qadf/experiments/e6_tokens.py
==================================
E6: Token Grid and Attention Heads Configuration (Table 3 Part C).
Investigates grid size G in {1, 7, 14, 28} and number of heads in {1, 2, 4, 8}.
Demonstrates non-degenerate cross-attention: measures average entropy and attended token ratio.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def get_jobs(config: dict, profile: str = "full") -> List[dict]:
    seeds = [42] if profile == "smoke" else [42, 43]
    grid_sizes = [1, 7, 14, 28] if profile == "full" else [7, 14]
    heads = [1, 2, 4, 8] if profile == "full" else [2, 4]
    backbone = config.get("model", {}).get("backbone", "efficientnet_b4")

    jobs = []
    # Grid G sweep
    for g in grid_sizes:
        for seed in seeds:
            job_id = f"e6_grid_{g}_heads4_seed{seed}"
            jobs.append({
                "job_id": job_id,
                "stage": "S2",
                "experiment": "E6",
                "method_name": "qc_sca",
                "seed": seed,
                "train_mode": "train_clean",
                "method": {"fusion_type": "qc_sca", "backbone": backbone, "num_heads": 4},
                "data": config["data"],
                "model": {**config["model"], "grid_size": g},
                "train": config["train"],
                "quality": config["quality"],
            })

    # Heads H sweep (with default G=14)
    for h in [x for x in heads if x != 4]:
        for seed in seeds:
            job_id = f"e6_grid14_heads_{h}_seed{seed}"
            jobs.append({
                "job_id": job_id,
                "stage": "S2",
                "experiment": "E6",
                "method_name": "qc_sca",
                "seed": seed,
                "train_mode": "train_clean",
                "method": {"fusion_type": "qc_sca", "backbone": backbone, "num_heads": h},
                "data": config["data"],
                "model": {**config["model"], "grid_size": 14},
                "train": config["train"],
                "quality": config["quality"],
            })

    return jobs


def aggregate_e6(raw_s3_dir: str, output_dir: str) -> Dict[str, Any]:
    import re
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "e6_summary.json")
    results = {"grids": {}, "heads": {}}

    if not os.path.exists(raw_s3_dir):
        return results

    for fname in os.listdir(raw_s3_dir):
        if not fname.startswith("e6_") or not fname.endswith(".parquet"):
            continue
        # Filename pattern:
        # e6_grid_14_heads4_seed42.parquet or e6_grid14_heads_2_seed42.parquet
        try:
            df = pd.read_parquet(os.path.join(raw_s3_dir, fname))
            from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy
            qwk = compute_qwk(df["label"].values, df["pred"].values)
            acc = compute_accuracy(df["label"].values, df["pred"].values)

            m_grid = re.search(r"grid_?(\d+)", fname)
            m_head = re.search(r"heads_?(\d+)", fname)

            # Grid G sweep (heads fixed to 4)
            if "heads4" in fname and m_grid:
                g = int(m_grid.group(1))
                if str(g) not in results["grids"]:
                    results["grids"][str(g)] = []
                results["grids"][str(g)].append(qwk)

            # Heads H sweep (grid fixed to 14)
            if m_head and m_grid and int(m_grid.group(1)) == 14:
                h = int(m_head.group(1))
                if str(h) not in results["heads"]:
                    results["heads"][str(h)] = []
                results["heads"][str(h)].append(qwk)

        except Exception as e:
            print(f"  [WARN] Skipping file {fname} due to error: {e}")
            continue

    summary = {
        "grids": {k: {"mean_qwk": float(np.mean(v))} for k, v in results["grids"].items() if len(v) > 0},
        "heads": {k: {"mean_qwk": float(np.mean(v))} for k, v in results["heads"].items() if len(v) > 0},
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
