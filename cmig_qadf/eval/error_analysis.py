"""
cmig_qadf/eval/error_analysis.py
================================
Experiment E10: Error case analysis and clinical case studies (Error Case Analysis).
1. Exports results/processed/e10_cases.csv:
   id, label, fusion prediction, single-view 1 & 2 predictions, q, q_hat, degradation type, attention mass.
2. Categorizes clinical cases:
   - Harm cases (Fusion worse than best single view)
   - Help cases (Fusion better than best single view)
   - Common failure cases (Both single-view and fusion incorrect)
3. Decomposes Harm Rate across ICDR DR grades (0 to 4).
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any


def analyze_errors_e10(
    df_fusion: pd.DataFrame,
    df_v1: pd.DataFrame,
    df_v2: pd.DataFrame,
    output_csv_path: str = "results/processed/e10_cases.csv",
) -> Dict[str, Any]:
    """
    Compare fusion predictions against single-view baselines and extract error cases.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Merge by sample_id
    m1 = df_v1[["sample_id", "pred"]].rename(columns={"pred": "pred_v1"})
    m2 = df_v2[["sample_id", "pred"]].rename(columns={"pred": "pred_v2"})

    merged = df_fusion.merge(m1, on="sample_id", how="left").merge(m2, on="sample_id", how="left")

    # Determine best single view per sample (view with higher effective quality score)
    merged["best_view_used"] = np.where(merged["q1_effective"] >= merged["q2_effective"], "v1", "v2")
    merged["pred_bestview"] = np.where(merged["best_view_used"] == "v1", merged["pred_v1"], merged["pred_v2"])

    err_f = np.abs(merged["pred"] - merged["label"])
    err_b = np.abs(merged["pred_bestview"] - merged["label"])

    merged["err_fusion"] = err_f
    merged["err_bestview"] = err_b

    merged["case_category"] = "tie"
    merged.loc[err_f > err_b, "case_category"] = "harm"
    merged.loc[err_f < err_b, "case_category"] = "help"
    merged.loc[(err_f > 0) & (err_b > 0), "case_category"] = "both_fail"

    # Export detailed CSV
    export_cols = [
        "sample_id", "patient_id", "eye_id", "label",
        "pred", "pred_v1", "pred_v2", "pred_bestview",
        "q1_human", "q2_human", "q1_effective", "q2_effective",
        "case_category", "err_fusion", "err_bestview"
    ]
    avail_cols = [c for c in export_cols if c in merged.columns]
    merged[avail_cols].to_csv(output_csv_path, index=False)
    print(f"  ✓ Exported E10 error case manifest → {output_csv_path}")

    # Harm Rate decomposed across ICDR grades (0 to 4)
    per_class_harm = {}
    for c in range(5):
        sub = merged[merged["label"] == c]
        if len(sub) > 0:
            h_rate = float(np.mean(sub["case_category"] == "harm"))
            per_class_harm[f"class_{c}_harm_rate"] = h_rate
        else:
            per_class_harm[f"class_{c}_harm_rate"] = 0.0

    return {
        "total_samples": len(merged),
        "harm_count": int(np.sum(merged["case_category"] == "harm")),
        "help_count": int(np.sum(merged["case_category"] == "help")),
        "tie_count": int(np.sum(merged["case_category"] == "tie")),
        "per_class_harm": per_class_harm,
        "cases_df": merged,
    }
