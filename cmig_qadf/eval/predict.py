"""
cmig_qadf/eval/predict.py
=========================
Stores per-sample predictions (Sample-Level Predictions) to Parquet (or CSV/NPZ fallback).
Supports paired Bootstrap statistics, quality gap stratification (E3), and error case analysis (E10).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from typing import Dict, Any, Optional

from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy, compute_auc, compute_per_class_kappa


def run_prediction(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: Optional[str] = None,
    degradation_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute inference across data loader and record complete sample-level predictions and diagnostics.
    """
    model.eval()
    records = []

    all_y_true = []
    all_y_pred = []
    all_y_probs = []

    with torch.no_grad():
        for batch in loader:
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            q1_raw = batch["q1"].to(device, non_blocking=True)
            q2_raw = batch["q2"].to(device, non_blocking=True)
            lbls = batch["label"]

            with autocast("cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                out = model(img1, img2, q1_raw, q2_raw)

            preds = out["pred"].cpu().numpy()
            probs = out["probs"].cpu().numpy()
            exp_grades = out["expected_grade"].cpu().numpy()
            logits = out["logits"].cpu().numpy()

            q1_eff = out["q1"].cpu().numpy().flatten()
            q2_eff = out["q2"].cpu().numpy().flatten()
            f_cons = out["f_cons"].cpu().numpy()

            # Attention diagnostic metrics if available
            attn_sink_mass = []
            if "diag12" in out and "sink_mass" in out["diag12"]:
                attn_sink_mass = [float(out["diag12"]["sink_mass"])] * len(lbls)
            else:
                attn_sink_mass = [0.0] * len(lbls)

            for i in range(len(lbls)):
                rec = {
                    "sample_id": batch["pair_id"][i],
                    "patient_id": str(batch["patient_id"][i]),
                    "eye_id": str(batch["eye_id"][i]),
                    "label": int(lbls[i]),
                    "pred": int(preds[i]),
                    "expected_grade": float(exp_grades[i]),
                    "q1_human": float(q1_raw[i].item()),
                    "q2_human": float(q2_raw[i].item()),
                    "q1_effective": float(q1_eff[i]),
                    "q2_effective": float(q2_eff[i]),
                    "q_gap_human": float(abs(q1_raw[i].item() - q2_raw[i].item())),
                    "q_gap_effective": float(abs(q1_eff[i] - q2_eff[i])),
                    "sink_mass": float(attn_sink_mass[i]),
                }
                # Logits and probability distributions
                for k in range(4):
                    rec[f"logit_{k}"] = float(logits[i, k])
                for c in range(5):
                    rec[f"prob_{c}"] = float(probs[i, c])

                # Degradation condition metadata if present
                if degradation_info:
                    rec["deg_type"] = degradation_info.get("type", "clean")
                    rec["deg_level"] = degradation_info.get("level", 0)
                    rec["deg_view"] = degradation_info.get("view", "none")

                records.append(rec)

            all_y_true.append(lbls.numpy())
            all_y_pred.append(preds)
            all_y_probs.append(probs)

    y_t = np.concatenate(all_y_true)
    y_p = np.concatenate(all_y_pred)
    y_pr = np.concatenate(all_y_probs)

    summary_metrics = {
        "QWK": compute_qwk(y_t, y_p),
        "Accuracy": compute_accuracy(y_t, y_p),
        "AUC": compute_auc(y_t, y_pr),
        **compute_per_class_kappa(y_t, y_p),
        "num_samples": len(y_t),
    }

    df = pd.DataFrame(records)

    # Save per-sample prediction table
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            df.to_parquet(output_path, index=False)
        except Exception:
            # Fallback if pyarrow / fastparquet is not installed
            csv_path = output_path.replace(".parquet", ".csv")
            df.to_csv(csv_path, index=False)

    return {
        "metrics": summary_metrics,
        "dataframe": df,
        "y_true": y_t,
        "y_pred": y_p,
        "y_probs": y_pr,
    }
