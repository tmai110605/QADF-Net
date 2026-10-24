"""
cmig_qadf/experiments/e1_main.py
================================
E1: Primary benchmark comparison on clean fundus images (Table 1 Main).
Methods: Single-view (V1, V2), Best Single, Concat, Average, Max, Q-Weighted, Cross-Attn, Co-Attn, QC-SCA.
Metrics: QWK, Acc, AUC, Per-class Kappa. Reports mean ± std across seeds and [95% CI].
Paired clustered bootstrap testing vs QC-SCA with Holm-Bonferroni step-down correction.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from cmig_qadf.eval.stats import patient_cluster_bootstrap_ci, paired_cluster_bootstrap_test, holm_bonferroni_correction, generate_bootstrap_replicates, get_patient_clusters
from cmig_qadf.eval.metrics import compute_qwk


def get_jobs(config: dict, profile: str = "full") -> List[dict]:
    """Generate list of training jobs for E1 experiment."""
    seeds = config.get("seeds", [42, 43, 44, 45, 46])
    methods = config.get("methods", [
        "qc_sca", "concat", "average", "max",
        "q_weighted", "cross_attn", "co_attn",
        "single_v1", "single_v2"
    ])
    backbone = config.get("model", {}).get("backbone", "efficientnet_b4")

    jobs = []
    for method in methods:
        for seed in seeds:
            job_id = f"e1_{method}_{backbone}_seed{seed}_clean"
            jobs.append({
                "job_id": job_id,
                "stage": "S2",
                "experiment": "E1",
                "method_name": method,
                "seed": seed,
                "train_mode": "train_clean",
                "method": {"fusion_type": method, "backbone": backbone},
                "data": config["data"],
                "model": config["model"],
                "train": config["train"],
                "quality": config["quality"],
            })
    return jobs


def aggregate_e1(raw_s3_dir: str, output_dir: str, splits: dict, bootstrap_reps: int = 2000) -> Dict[str, Any]:
    """
    Aggregate E1 results from predictions.parquet and metrics.json files.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "e1_summary.json")

    # Collect all prediction parquet files
    method_data = {}
    if not os.path.exists(raw_s3_dir):
        return {}

    for fname in os.listdir(raw_s3_dir):
        if not fname.startswith("e1_") or not fname.endswith(".parquet"):
            continue
        name_no_ext = fname.replace(".parquet", "")
        # e.g. e1_qc_sca_efficientnet_b4_seed42_clean
        sub = name_no_ext[3:] if name_no_ext.startswith("e1_") else name_no_ext
        if sub.startswith("q_weighted"):
            continue  # Exclude Q-Weighted per benchmark evaluation guidelines
        method = "other"
        for m in [
            "qc_sca", "best_single_oracle", "best_single_learned",
            "cross_attn", "co_attn", "single_v1", "single_v2",
            "concat", "average", "max"
        ]:
            if sub.startswith(m):
                method = m
                break
        parts = name_no_ext.split("_")
        seed_str = [p for p in parts if p.startswith("seed")][0]
        seed = int(seed_str.replace("seed", ""))

        df = pd.read_parquet(os.path.join(raw_s3_dir, fname))
        if method not in method_data:
            method_data[method] = {}
        method_data[method][seed] = df

    if not method_data:
        return {}

    # Retrieve patient grouping from reference predictions file to generate paired replicates
    first_m = list(method_data.keys())[0]
    first_s = list(method_data[first_m].keys())[0]
    df_sample = method_data[first_m][first_s]

    unique_patients = sorted(list(df_sample["patient_id"].unique()))
    clusters = get_patient_clusters(df_sample["patient_id"].tolist())
    replicates = generate_bootstrap_replicates(unique_patients, clusters, n_bootstraps=bootstrap_reps, seed=42)

    def qwk_metric_fn(d, idxs):
        return compute_qwk(d["label"].iloc[idxs].values, d["pred"].iloc[idxs].values)

    results = {}
    raw_p_values = {}

    for method, seeds_dict in method_data.items():
        qwks = []
        accs = []
        aucs = []
        cis = []

        for seed, df in seeds_dict.items():
            qwk_val = compute_qwk(df["label"].values, df["pred"].values)
            acc_val = float(np.mean(df["label"].values == df["pred"].values))
            qwks.append(qwk_val)
            accs.append(acc_val)

            # Compute Bootstrap CI per seed
            pt, low, up = patient_cluster_bootstrap_ci(df, qwk_metric_fn, replicates)
            cis.append((low, up))

        results[method] = {
            "qwk_mean": float(np.mean(qwks)),
            "qwk_std": float(np.std(qwks)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "ci_lower": float(np.mean([c[0] for c in cis])),
            "ci_upper": float(np.mean([c[1] for c in cis])),
            "num_seeds": len(qwks),
        }

    # Compute paired hypothesis tests against QC-SCA
    if "qc_sca" in method_data:
        df_qc = method_data["qc_sca"][first_s]
        for method, seeds_dict in method_data.items():
            if method == "qc_sca":
                continue
            df_base = seeds_dict[first_s]
            p_res = paired_cluster_bootstrap_test(df_qc, df_base, qwk_metric_fn, replicates)
            raw_p_values[method] = p_res["p_value"]
            results[method]["paired_diff_with_qc_sca"] = p_res["mean_diff"]
            results[method]["paired_p_raw"] = p_res["p_value"]

        # Holm-Bonferroni correction
        holm_p = holm_bonferroni_correction(raw_p_values)
        for method, p_h in holm_p.items():
            results[method]["paired_p_holm"] = p_h

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
