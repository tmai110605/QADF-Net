"""
cmig_qadf/experiments/e10_errors.py
===================================
E10: Error case analysis and clinical case library extraction (Figure 6 & e10_cases.csv).
Compares fusion predictions against single-view 1 & 2 to isolate harm / help / failure cases.
"""

import os
import json
import pandas as pd
from typing import Dict, Any

from cmig_qadf.eval.error_analysis import analyze_errors_e10


def run_e10_experiment(
    raw_s3_dir: str,
    output_dir: str = "results/processed"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "e10_cases.csv")
    out_json = os.path.join(output_dir, "e10_summary.json")

    # Locate parquet predictions for QC-SCA, single_v1, and single_v2
    files = [f for f in os.listdir(raw_s3_dir) if f.endswith(".parquet")] if os.path.exists(raw_s3_dir) else []
    qc_files = [f for f in files if "qc_sca" in f]
    v1_files = [f for f in files if "single_v1" in f]
    v2_files = [f for f in files if "single_v2" in f]

    if not qc_files or not v1_files or not v2_files:
        return {}

    df_qc = pd.read_parquet(os.path.join(raw_s3_dir, qc_files[0]))
    df_v1 = pd.read_parquet(os.path.join(raw_s3_dir, v1_files[0]))
    df_v2 = pd.read_parquet(os.path.join(raw_s3_dir, v2_files[0]))

    res = analyze_errors_e10(df_qc, df_v1, df_v2, output_csv_path=out_csv)
    
    summary = {
        "total_samples": res["total_samples"],
        "harm_count": res["harm_count"],
        "help_count": res["help_count"],
        "tie_count": res["tie_count"],
        "per_class_harm": res["per_class_harm"],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
