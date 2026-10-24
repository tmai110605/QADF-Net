"""
cmig_qadf/experiments/e3_strata.py
==================================
E3: Stratification by natural quality gap |q1 - q2| (Table 2).
Divides samples into tertiles (Low, Medium, High Gap) using both human q and learned q_hat on clean data.
Computes QWK, Fusion Gain, and Harm Rate for each stratum.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from cmig_qadf.eval.metrics import compute_qwk
from cmig_qadf.eval.new_metrics import compute_fusion_gain, compute_harm_rate


def aggregate_e3(raw_s3_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Analyze |q1 - q2| tertiles from predictions.parquet files across E1 models.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "e3_summary.json")

    results = {}
    if not os.path.exists(raw_s3_dir):
        return results

    # Locate prediction files for QC-SCA and single-views
    files = [f for f in os.listdir(raw_s3_dir) if f.endswith(".parquet")]
    qc_files = [f for f in files if "qc_sca" in f]
    v1_files = [f for f in files if "single_v1" in f]
    v2_files = [f for f in files if "single_v2" in f]
    q_avg_files = [f for f in files if "q_weighted" in f]

    if not qc_files:
        return results

    df_qc = pd.read_parquet(os.path.join(raw_s3_dir, qc_files[0]))
    df_v1 = pd.read_parquet(os.path.join(raw_s3_dir, v1_files[0])) if v1_files else None
    df_v2 = pd.read_parquet(os.path.join(raw_s3_dir, v2_files[0])) if v2_files else None
    df_qavg = pd.read_parquet(os.path.join(raw_s3_dir, q_avg_files[0])) if q_avg_files else None

    # Gap tertiles
    gaps_human = df_qc["q_gap_human"].values
    gaps_eff   = df_qc["q_gap_effective"].values

    tertiles_h = np.percentile(gaps_human, [33.3, 66.7])
    tertiles_e = np.percentile(gaps_eff, [33.3, 66.7])

    for gap_type, tertiles, gaps in [
        ("human_q", tertiles_h, gaps_human),
        ("learned_qhat", tertiles_e, gaps_eff)
    ]:
        results[gap_type] = {}
        bins = [
            ("Low Gap", gaps <= tertiles[0]),
            ("Medium Gap", (gaps > tertiles[0]) & (gaps <= tertiles[1])),
            ("High Gap", gaps > tertiles[1]),
        ]

        for stratum_name, mask in bins:
            sub_qc = df_qc[mask]
            sub_qwk_qc = compute_qwk(sub_qc["label"].values, sub_qc["pred"].values)

            # Single view comparison
            sub_qwk_v1 = compute_qwk(df_v1[mask]["label"].values, df_v1[mask]["pred"].values) if df_v1 is not None else 0.0
            sub_qwk_v2 = compute_qwk(df_v2[mask]["label"].values, df_v2[mask]["pred"].values) if df_v2 is not None else 0.0
            best_single = max(sub_qwk_v1, sub_qwk_v2)

            sub_qwk_qavg = compute_qwk(df_qavg[mask]["label"].values, df_qavg[mask]["pred"].values) if df_qavg is not None else 0.0

            # Harm rate per sample
            if df_v1 is not None and df_v2 is not None:
                pred_best = np.where(sub_qc["q1_effective"] >= sub_qc["q2_effective"], df_v1[mask]["pred"].values, df_v2[mask]["pred"].values)
                harm_info = compute_harm_rate(sub_qc["label"].values, sub_qc["pred"].values, pred_best)
            else:
                harm_info = {"harm_rate": 0.0, "tie_rate": 1.0, "help_rate": 0.0}

            results[gap_type][stratum_name] = {
                "num_samples": int(np.sum(mask)),
                "qwk_qc_sca": float(sub_qwk_qc),
                "qwk_q_weighted": float(sub_qwk_qavg),
                "qwk_best_single": float(best_single),
                "fusion_gain": float(compute_fusion_gain(sub_qwk_qc, best_single)),
                **harm_info
            }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
