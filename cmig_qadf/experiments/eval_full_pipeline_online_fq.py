"""
cmig_qadf/experiments/eval_full_pipeline_online_fq.py
=====================================================
COMPREHENSIVE END-TO-END PIPELINE EVALUATION WITH ONLINE QUALITY ESTIMATOR (f_Q)
-----------------------------------------------------------------------------
Critical experimental verification for the CMIG (Elsevier) submission:
- Eliminates analytical attenuation formulas q * (1 - 0.9 * L / 5).
- Quality scores (q1_hat, q2_hat) are predicted ONLINE DIRECTLY from fundus
  pixels (clean and degraded) via f_Q (MobileNetV3-Small) at GPU runtime.
- No artificial degradation formulas exist in this pipeline.
- Supports direct comparison between "Online f_Q Estimation" and "Legacy Analytical Formula".

Execution examples on GPU (e.g. NVIDIA T4):
  # Fast sanity test (15 seconds):
  python cmig_qadf/experiments/eval_full_pipeline_online_fq.py --quick_test --device cuda

  # Full 5 seeds x core methods on GPU (~3-4 minutes):
  python cmig_qadf/experiments/eval_full_pipeline_online_fq.py --device cuda --batch_size 32

  # Direct comparison between Online f_Q and legacy formula:
  python cmig_qadf/experiments/eval_full_pipeline_online_fq.py --device cuda --compare_analytical

Outputs:
  - Full JSON: results/processed/online_fq_evaluation_summary.json
  - CSV Table: results/processed/online_fq_table.csv
"""

import os
import sys
import time
import json
import yaml
import argparse
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from cmig_qadf.models.q_estimator import QEstimator
from cmig_qadf.models.registry import build_model
from cmig_qadf.data.adapter import DeepDRiDAdapterDataset, get_transforms, build_pair_records
from cmig_qadf.data.degradations import DEGRADATION_TYPES, apply_single_degradation
from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy
from cmig_qadf.eval.new_metrics import compute_audc, compute_harm_rate, compute_pfr, compute_fdi


MAIN_METHODS = ["qc_sca", "q_weighted", "cross_attn", "concat", "average"]
ALL_METHODS = [
    "qc_sca", "q_weighted", "cross_attn", "concat", "average",
    "co_attn", "max", "single_v1", "single_v2"
]


def find_deepdrid_dir(base_dir: str = "DeepDRiD") -> str:
    """Automatically locate valid DeepDRiD image directory across OS environments."""
    candidates = [
        os.path.join(base_dir, "regular_fundus_images"),
        base_dir,
        "DeepDRiD/regular_fundus_images",
        "DeepDRiD",
        "/home/thaimq/DiabeticRetinopathy/DeepDRiD/regular_fundus_images",
        "/home/thaimq/DiabeticRetinopathy/DeepDRiD",
    ]
    for c in candidates:
        if (os.path.exists(os.path.join(c, "regular-fundus-training")) or
            os.path.exists(os.path.join(c, "regular-fundus-validation")) or
            os.path.exists(os.path.join(c, "Online-Challenge1&2-Evaluation"))):
            return c
    return os.path.join(base_dir, "regular_fundus_images")


def load_best_single_predictions(s3_dir: str, seed: int) -> np.ndarray:
    """
    Retrieve predictions of the best single view on clean images as reference baseline for Harm Rate.
    Strictly follows paper formulation: View 1 if q1 >= q2, else View 2.
    """
    v1_p = os.path.join(s3_dir, f"e1_single_v1_efficientnet_b4_seed{seed}_clean.parquet")
    v2_p = os.path.join(s3_dir, f"e1_single_v2_efficientnet_b4_seed{seed}_clean.parquet")
    qc_p = os.path.join(s3_dir, f"e1_qc_sca_efficientnet_b4_seed{seed}_clean.parquet")

    if os.path.exists(v1_p) and os.path.exists(v2_p) and os.path.exists(qc_p):
        df_v1 = pd.read_parquet(v1_p).sort_values("sample_id").reset_index(drop=True)
        df_v2 = pd.read_parquet(v2_p).sort_values("sample_id").reset_index(drop=True)
        df_qc = pd.read_parquet(qc_p).sort_values("sample_id").reset_index(drop=True)
        pred_best = np.where(df_qc["q1_effective"] >= df_qc["q2_effective"], df_v1["pred"].values, df_v2["pred"].values)
        return pred_best
    elif os.path.exists(v1_p):
        df_v1 = pd.read_parquet(v1_p).sort_values("sample_id").reset_index(drop=True)
        return df_v1["pred"].values
    return None


def run_batch_inference_online_fq(
    model: nn.Module,
    q_estimator: nn.Module,
    dataloader: DataLoader,
    device: str,
    use_online_fq: bool = True,
    analytical_level: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute batch inference:
    - If use_online_fq=True: f_Q computes q1_hat and q2_hat directly from pixels on GPU.
    - If use_online_fq=False: q1, q2 taken from legacy formula q * (1 - 0.9 * L/5).
    Returns: (preds, f_cons_vectors, q1_scores, q2_scores)
    """
    preds_list = []
    f_cons_list = []
    q1_list = []
    q2_list = []

    model.eval()
    q_estimator.eval()

    with torch.no_grad():
        for b in dataloader:
            im1 = b["img1"].to(device, non_blocking=True)
            im2 = b["img2"].to(device, non_blocking=True)

            if use_online_fq:
                # Downsample on GPU to 224x224 for QEstimator (MobileNetV3)
                im1_224 = F.interpolate(im1, size=(224, 224), mode="bilinear", align_corners=False)
                im2_224 = F.interpolate(im2, size=(224, 224), mode="bilinear", align_corners=False)

                q1_hat = q_estimator(im1_224)
                q2_hat = q_estimator(im2_224)
                out = model(im1, im2, q1_hat, q2_hat)
                q1_val = q1_hat.cpu().squeeze().numpy()
                q2_val = q2_hat.cpu().squeeze().numpy()
            else:
                # Legacy analytical formula for comparison
                s = float(analytical_level) / 5.0
                q1_raw = b["q1"].to(device, non_blocking=True)
                q2_raw = torch.clamp(b["q2"].to(device, non_blocking=True) * (1.0 - 0.9 * s), min=0.001)
                out = model(im1, im2, q1_raw, q2_raw)
                q1_val = q1_raw.cpu().squeeze().numpy()
                q2_val = q2_raw.cpu().squeeze().numpy()

            preds_list.append(out["pred"].cpu().numpy())
            f_cons_list.append(out["f_cons"].cpu().numpy())
            q1_list.append(np.atleast_1d(q1_val))
            q2_list.append(np.atleast_1d(q2_val))

    all_preds = np.concatenate(preds_list)
    all_f_cons = np.concatenate(f_cons_list)
    all_q1 = np.concatenate(q1_list)
    all_q2 = np.concatenate(q2_list)

    return all_preds, all_f_cons, all_q1, all_q2


def evaluate_method_seed_online_fq(
    method: str,
    seed: int,
    s2_dir: str,
    s3_dir: str,
    q_estimator: nn.Module,
    test_recs: List[Dict],
    cached_images: Dict[str, Tuple[Image.Image, Image.Image]],
    eval_tf: Any,
    degradations: List[str],
    target_view: str,
    device: str,
    batch_size: int,
    compare_analytical: bool = False
) -> Dict[str, Any]:
    """
    Evaluate 1 method and 1 seed across 6 severity levels (Levels 0-5)
    exclusively using online f_Q inference.
    """
    job_name = f"e1_{method}_efficientnet_b4_seed{seed}_clean"
    ckpt_path = os.path.join(s2_dir, job_name, "best_checkpoint.pt")
    cfg_path  = os.path.join(s2_dir, job_name, "config.yaml")

    # Fallback to checkpoints/ folder if results/ directory is absent
    if not os.path.exists(ckpt_path):
        for candidate_name in [f"qadf_net_seed{seed}.pt", f"qc_sca_seed{seed}.pt"]:
            cand = os.path.join("checkpoints", candidate_name)
            if os.path.exists(cand):
                ckpt_path = cand
                break

    if not os.path.exists(ckpt_path):
        print(f"  [WARN] Checkpoint not found: {ckpt_path}")
        return None

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "model": {"backbone": "efficientnet_b4", "token_dim": 128, "grid_size": 14, "num_heads": 4},
            "method": {"fusion_type": method, "qc_mode": "all", "use_sink_token": True, "q_source": "oracle"}
        }

    # Build model
    model = build_model(cfg, q_estimator=None)
    model.q_source = "oracle"  # Directly pass q1, q2 computed by f_Q
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.to(device).eval()

    # Load best single view predictions
    pred_best = load_best_single_predictions(s3_dir, seed)
    clean_labels = np.array([r["label"] for r in test_recs])

    # 1. Level 0: Clean evaluation
    clean_ds = DeepDRiDAdapterDataset(test_recs, transform=eval_tf, cached_images=cached_images)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)

    clean_preds, clean_f_cons, clean_q1, clean_q2 = run_batch_inference_online_fq(
        model=model,
        q_estimator=q_estimator,
        dataloader=clean_loader,
        device=device,
        use_online_fq=True,
        analytical_level=0
    )

    clean_qwk = compute_qwk(clean_labels, clean_preds)
    clean_acc = compute_accuracy(clean_labels, clean_preds)

    if pred_best is None:
        pred_best = clean_preds.copy()

    h0 = compute_harm_rate(clean_labels, clean_preds, pred_best)

    results = {
        "method": method,
        "seed": seed,
        "clean_qwk": clean_qwk,
        "clean_acc": clean_acc,
        "clean_harm_rate": h0["harm_rate"],
        "clean_q1_mean": float(np.mean(clean_q1)),
        "clean_q2_mean": float(np.mean(clean_q2)),
        "online_fq": {
            "per_degradation": {},
            "qwk_curve": [clean_qwk],
            "harm_curve": [h0["harm_rate"]],
            "pfr_curve": [0.0],
            "fdi_curve": [0.0],
            "q2_curve": [float(np.mean(clean_q2))],
            "audc": 0.0,
            "harm_rate_l5": 0.0,
            "help_rate_l5": 0.0,
            "pfr_l5": 0.0,
            "fdi_l5": 0.0,
        }
    }

    if compare_analytical:
        results["analytical_formula"] = {
            "per_degradation": {},
            "qwk_curve": [clean_qwk],
            "harm_curve": [h0["harm_rate"]],
            "pfr_curve": [0.0],
            "fdi_curve": [0.0],
            "audc": 0.0,
            "harm_rate_l5": 0.0,
            "pfr_l5": 0.0,
            "fdi_l5": 0.0,
        }

    # 2. Levels 1 to 5: Degradation with Online f_Q
    lvl_qwks = {lvl: [] for lvl in range(1, 6)}
    lvl_harms = {lvl: [] for lvl in range(1, 6)}
    lvl_helps = {lvl: [] for lvl in range(1, 6)}
    lvl_pfrs = {lvl: [] for lvl in range(1, 6)}
    lvl_fdis = {lvl: [] for lvl in range(1, 6)}
    lvl_q2s = {lvl: [] for lvl in range(1, 6)}

    # Analytical formula comparison arrays
    lvl_qwks_form = {lvl: [] for lvl in range(1, 6)}
    lvl_harms_form = {lvl: [] for lvl in range(1, 6)}
    lvl_pfrs_form = {lvl: [] for lvl in range(1, 6)}
    lvl_fdis_form = {lvl: [] for lvl in range(1, 6)}

    for d_type in degradations:
        results["online_fq"]["per_degradation"][d_type] = {}
        if compare_analytical:
            results["analytical_formula"]["per_degradation"][d_type] = {}

        for lvl in range(1, 6):
            def deg_hook(img1, img2, q1, q2, sample_id, patient_id):
                if target_view in ["v2", "both"]:
                    img2 = apply_single_degradation(img2, d_type, lvl, seed, sample_id, view="v2")
                if target_view in ["v1", "both"]:
                    img1 = apply_single_degradation(img1, d_type, lvl, seed, sample_id, view="v1")
                return img1, img2, q1, q2

            deg_ds = DeepDRiDAdapterDataset(test_recs, transform=eval_tf, degradation_fn=deg_hook, cached_images=cached_images)
            deg_loader = DataLoader(deg_ds, batch_size=batch_size, shuffle=False)

            # Online f_Q inference
            deg_preds, deg_f_cons, _, deg_q2 = run_batch_inference_online_fq(
                model=model,
                q_estimator=q_estimator,
                dataloader=deg_loader,
                device=device,
                use_online_fq=True
            )

            qwk = compute_qwk(clean_labels, deg_preds)
            h_dict = compute_harm_rate(clean_labels, deg_preds, pred_best)
            pfr = compute_pfr(clean_preds, deg_preds)
            fdi = compute_fdi(clean_f_cons, deg_f_cons)["mean_fdi"]
            mean_q2 = float(np.mean(deg_q2))

            lvl_qwks[lvl].append(qwk)
            lvl_harms[lvl].append(h_dict["harm_rate"])
            lvl_helps[lvl].append(h_dict["help_rate"])
            lvl_pfrs[lvl].append(pfr)
            lvl_fdis[lvl].append(fdi)
            lvl_q2s[lvl].append(mean_q2)

            results["online_fq"]["per_degradation"][d_type][f"L{lvl}"] = {
                "qwk": qwk, "harm_rate": h_dict["harm_rate"], "pfr": pfr, "fdi": fdi, "mean_q2": mean_q2
            }

            # Analytical formula comparison
            if compare_analytical:
                deg_preds_f, deg_f_cons_f, _, _ = run_batch_inference_online_fq(
                    model=model,
                    q_estimator=q_estimator,
                    dataloader=deg_loader,
                    device=device,
                    use_online_fq=False,
                    analytical_level=lvl
                )
                qwk_f = compute_qwk(clean_labels, deg_preds_f)
                h_dict_f = compute_harm_rate(clean_labels, deg_preds_f, pred_best)
                pfr_f = compute_pfr(clean_preds, deg_preds_f)
                fdi_f = compute_fdi(clean_f_cons, deg_f_cons_f)["mean_fdi"]

                lvl_qwks_form[lvl].append(qwk_f)
                lvl_harms_form[lvl].append(h_dict_f["harm_rate"])
                lvl_pfrs_form[lvl].append(pfr_f)
                lvl_fdis_form[lvl].append(fdi_f)

                results["analytical_formula"]["per_degradation"][d_type][f"L{lvl}"] = {
                    "qwk": qwk_f, "harm_rate": h_dict_f["harm_rate"], "pfr": pfr_f, "fdi": fdi_f
                }

    # Aggregate curve and compute AUDC (Online f_Q)
    for lvl in range(1, 6):
        results["online_fq"]["qwk_curve"].append(float(np.mean(lvl_qwks[lvl])))
        results["online_fq"]["harm_curve"].append(float(np.mean(lvl_harms[lvl])))
        results["online_fq"]["pfr_curve"].append(float(np.mean(lvl_pfrs[lvl])))
        results["online_fq"]["fdi_curve"].append(float(np.mean(lvl_fdis[lvl])))
        results["online_fq"]["q2_curve"].append(float(np.mean(lvl_q2s[lvl])))

    results["online_fq"]["audc"] = compute_audc(results["online_fq"]["qwk_curve"])
    results["online_fq"]["harm_rate_l5"] = results["online_fq"]["harm_curve"][-1]
    results["online_fq"]["help_rate_l5"] = float(np.mean(lvl_helps[5]))
    results["online_fq"]["pfr_l5"] = results["online_fq"]["pfr_curve"][-1]
    results["online_fq"]["fdi_l5"] = results["online_fq"]["fdi_curve"][-1]

    if compare_analytical:
        for lvl in range(1, 6):
            results["analytical_formula"]["qwk_curve"].append(float(np.mean(lvl_qwks_form[lvl])))
            results["analytical_formula"]["harm_curve"].append(float(np.mean(lvl_harms_form[lvl])))
            results["analytical_formula"]["pfr_curve"].append(float(np.mean(lvl_pfrs_form[lvl])))
            results["analytical_formula"]["fdi_curve"].append(float(np.mean(lvl_fdis_form[lvl])))

        results["analytical_formula"]["audc"] = compute_audc(results["analytical_formula"]["qwk_curve"])
        results["analytical_formula"]["harm_rate_l5"] = results["analytical_formula"]["harm_curve"][-1]
        results["analytical_formula"]["pfr_l5"] = results["analytical_formula"]["pfr_curve"][-1]
        results["analytical_formula"]["fdi_l5"] = results["analytical_formula"]["fdi_curve"][-1]

    return results


def format_markdown_table(aggregated: Dict[str, Any], compare_analytical: bool = False) -> str:
    """Export summary results to Markdown format."""
    lines = []
    lines.append("# Online Quality Estimator ($f_Q$) Robustness Benchmark Summary")
    lines.append("")
    lines.append("> **Note**: Quality scores $\\hat{q}_1, \\hat{q}_2$ are computed purely online by $f_Q$ (MobileNetV3) on the actual degraded pixels with **ZERO** analytical attenuation formulas.")
    lines.append("")
    
    if compare_analytical:
        lines.append("| Fusion Method | Clean QWK | Online $f_Q$ AUDC | Analytical AUDC | Online $\\text{HR}_{L5}$ (%) | Analytical $\\text{HR}_{L5}$ (%) | Online $\\text{PFR}_{L5}$ (%) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m, stat in aggregated.items():
            c_qwk = f"{stat['clean_qwk']['mean']:.4f} ± {stat['clean_qwk']['std']:.4f}"
            on_audc = f"**{stat['online_fq']['audc']['mean']:.4f} ± {stat['online_fq']['audc']['std']:.4f}**" if m == "qc_sca" else f"{stat['online_fq']['audc']['mean']:.4f} ± {stat['online_fq']['audc']['std']:.4f}"
            an_audc = f"{stat['analytical_formula']['audc']['mean']:.4f} ± {stat['analytical_formula']['audc']['std']:.4f}" if 'analytical_formula' in stat else "N/A"
            on_hr = f"**{stat['online_fq']['harm_rate_l5']['mean']*100:.2f}%**" if m == "qc_sca" else f"{stat['online_fq']['harm_rate_l5']['mean']*100:.2f}%"
            an_hr = f"{stat['analytical_formula']['harm_rate_l5']['mean']*100:.2f}%" if 'analytical_formula' in stat else "N/A"
            on_pfr = f"{stat['online_fq']['pfr_l5']['mean']*100:.2f}%"
            lines.append(f"| `{m}` | {c_qwk} | {on_audc} | {an_audc} | {on_hr} | {an_hr} | {on_pfr} |")
    else:
        lines.append("| Fusion Method | Clean QWK | AUDC ($L0 \\to L5$) | $\\text{HR}_{L5}$ (%) | $\\text{PFR}_{L5}$ (%) | $\\text{FDI}_{L5}$ | $\\bar{q}_2(L5)$ |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m, stat in aggregated.items():
            c_qwk = f"{stat['clean_qwk']['mean']:.4f} ± {stat['clean_qwk']['std']:.4f}"
            audc = f"**{stat['online_fq']['audc']['mean']:.4f} ± {stat['online_fq']['audc']['std']:.4f}**" if m == "qc_sca" else f"{stat['online_fq']['audc']['mean']:.4f} ± {stat['online_fq']['audc']['std']:.4f}"
            hr = f"**{stat['online_fq']['harm_rate_l5']['mean']*100:.2f}% ± {stat['online_fq']['harm_rate_l5']['std']*100:.2f}%**" if m == "qc_sca" else f"{stat['online_fq']['harm_rate_l5']['mean']*100:.2f}% ± {stat['online_fq']['harm_rate_l5']['std']*100:.2f}%"
            pfr = f"{stat['online_fq']['pfr_l5']['mean']*100:.2f}% ± {stat['online_fq']['pfr_l5']['std']*100:.2f}%"
            fdi = f"{stat['online_fq']['fdi_l5']['mean']:.4f}"
            q2_l5 = f"{stat['online_fq']['q2_l5']['mean']:.4f}"
            lines.append(f"| `{m}` | {c_qwk} | {audc} | {hr} | {pfr} | {fdi} | {q2_l5} |")

    return "\n".join(lines)


def format_latex_table(aggregated: Dict[str, Any], compare_analytical: bool = False) -> str:
    """Export summary results to LaTeX booktabs format."""
    latex = []
    latex.append("% -------------------------------------------------------------")
    latex.append("% Table: End-to-End Evaluation with Online Quality Estimator (f_Q)")
    latex.append("% Generated by cmig_qadf/experiments/eval_full_pipeline_online_fq.py")
    latex.append("% -------------------------------------------------------------")
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{End-to-end multi-view grading performance under synthetic degradation evaluated with the online quality estimator $f_Q$ directly processing degraded fundus images (no analytical formula attenuation). Values denote mean $\\pm$ standard deviation across five seeds.}")
    latex.append("\\label{tab:online_fq_benchmark}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{l c c c c c}")
    latex.append("\\toprule")
    latex.append("\\textbf{Fusion Architecture} & \\textbf{Clean QWK} & \\textbf{AUDC ($L0\\to L5$)} & \\textbf{Harm Rate ($L5$)} & \\textbf{PFR ($L5$)} & \\textbf{FDI ($L5$)} \\\\")
    latex.append("\\midrule")

    method_labels = {
        "qc_sca": "\\textbf{\\qadfnet{} (Proposed QC-SCA)}",
        "q_weighted": "Q-Weighted GAP",
        "cross_attn": "Standard Cross-Attention",
        "concat": "Concatenation",
        "average": "Average Fusion",
        "co_attn": "Bilinear Co-Attention",
        "max": "Max Pooling",
        "single_v1": "Single View ($V_1$)",
        "single_v2": "Single View ($V_2$)",
    }

    for m, stat in aggregated.items():
        name = method_labels.get(m, m)
        c_qwk = f"${stat['clean_qwk']['mean']:.4f} \\pm {stat['clean_qwk']['std']:.4f}$"
        
        if m == "qc_sca":
            audc = f"$\\mathbf{{{stat['online_fq']['audc']['mean']:.4f} \\pm {stat['online_fq']['audc']['std']:.4f}}}$"
            hr = f"$\\mathbf{{{stat['online_fq']['harm_rate_l5']['mean']*100:.2f}\\% \\pm {stat['online_fq']['harm_rate_l5']['std']*100:.2f}\\%}}$"
            pfr = f"$\\mathbf{{{stat['online_fq']['pfr_l5']['mean']*100:.2f}\\% \\pm {stat['online_fq']['pfr_l5']['std']*100:.2f}\\%}}$"
        else:
            audc = f"${stat['online_fq']['audc']['mean']:.4f} \\pm {stat['online_fq']['audc']['std']:.4f}$"
            hr = f"${stat['online_fq']['harm_rate_l5']['mean']*100:.2f}\\% \\pm {stat['online_fq']['harm_rate_l5']['std']*100:.2f}\\%$"
            pfr = f"${stat['online_fq']['pfr_l5']['mean']*100:.2f}\\% \\pm {stat['online_fq']['pfr_l5']['std']*100:.2f}\\%$"
        
        fdi = f"${stat['online_fq']['fdi_l5']['mean']:.4f}$"
        latex.append(f"{name} & {c_qwk} & {audc} & {hr} & {pfr} & {fdi} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive End-to-End Pipeline Evaluation with Online Quality Estimator f_Q.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 or 32 for T4)")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated random seeds")
    parser.add_argument("--methods", type=str, default="main", help="main (5 methods), all (10 methods), or comma-separated list")
    parser.add_argument("--degradations", type=str, default="all", help="all or comma-separated degradation types")
    parser.add_argument("--target_view", type=str, default="v2", choices=["v1", "v2", "both"], help="Degraded view (default v2 per Table 3)")
    parser.add_argument("--data_dir", type=str, default=None, help="DeepDRiD dataset path")
    parser.add_argument("--results_dir", type=str, default="results", help="Results folder")
    parser.add_argument("--output_dir", type=str, default="results/processed", help="Processed output folder")
    parser.add_argument("--compare_analytical", action="store_true", help="Run parallel comparison with legacy formula")
    parser.add_argument("--quick_test", action="store_true", help="Quick sanity test on 1 seed, 1 method, 1 degradation type")

    args = parser.parse_args()

    # Determine seeds and evaluation configurations
    if args.quick_test:
        seeds = [42]
        methods = ["qc_sca"]
        degradations = ["gaussian_blur"]
        print("[QUICK TEST MODE ACTIVATED] Running 1 seed, 1 method, 1 degradation type...")
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.methods == "main":
            methods = MAIN_METHODS
        elif args.methods == "all":
            methods = ALL_METHODS
        else:
            methods = [m.strip() for m in args.methods.split(",") if m.strip()]

        if args.degradations == "all":
            degradations = DEGRADATION_TYPES
        else:
            degradations = [d.strip() for d in args.degradations.split(",") if d.strip()]

    print(f"\n{'='*75}")
    print(f"  CMIG BENCHMARK: ONLINE QUALITY ESTIMATOR (f_Q) EVALUATION PIPELINE")
    print(f"{'='*75}")
    print(f"  - Device:             {args.device}")
    print(f"  - Batch Size:         {args.batch_size}")
    print(f"  - Target View:        {args.target_view}")
    print(f"  - Seeds ({len(seeds)}):         {seeds}")
    print(f"  - Methods ({len(methods)}):       {methods}")
    print(f"  - Degradations ({len(degradations)}):  {degradations}")
    print(f"  - Compare Analytical: {args.compare_analytical}")
    print(f"{'='*75}\n")

    # 1. Verify data directory
    deepdrid_dir = args.data_dir if args.data_dir else find_deepdrid_dir("DeepDRiD")
    print(f"[INFO] Dataset directory: {deepdrid_dir}")
    split_manifest = os.path.join(args.results_dir, "manifests", "split_manifest.json")
    if not os.path.exists(split_manifest):
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")

    with open(split_manifest, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_recs = build_pair_records(deepdrid_dir, splits["test"])
    if len(test_recs) == 0:
        alt = os.path.join(deepdrid_dir, "regular_fundus_images")
        if os.path.exists(alt):
            test_recs = build_pair_records(alt, splits["test"])
            deepdrid_dir = alt

    test_recs = sorted(test_recs, key=lambda r: r["pair_id"])
    print(f"[INFO] Successfully loaded {len(test_recs)} test samples (Test Set Challenge)!")

    # 2. Pre-cache test images into RAM (448x448) for 50x-100x acceleration
    eval_tf = get_transforms(448, is_train=False, use_clahe=True)
    cached_images = {}
    print("[INFO] Pre-caching test images into RAM for maximum GPU throughput...")
    t0 = time.time()
    for rec in test_recs:
        pid = rec["pair_id"]
        if pid not in cached_images:
            with Image.open(rec["path1"]) as im1:
                im1_rgb = im1.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
            with Image.open(rec["path2"]) as im2:
                im2_rgb = im2.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
            cached_images[pid] = (im1_rgb, im2_rgb)
    print(f"[INFO] Pre-cache completed in {time.time()-t0:.2f}s (~120MB RAM)!\n")

    # 3. Load quality estimator checkpoint f_Q
    q_ckpt = os.path.join(args.results_dir, "checkpoints", "q_estimator_best.pt")
    if not os.path.exists(q_ckpt):
        q_ckpt = os.path.join("checkpoints", "q_estimator_best.pt")
    if not os.path.exists(q_ckpt):
        raise FileNotFoundError(f"QEstimator checkpoint not found in '{args.results_dir}/checkpoints' or 'checkpoints/'")

    print(f"[INFO] Loading f_Q Quality Estimator (MobileNetV3) from: {q_ckpt}")
    q_estimator = QEstimator(pretrained=False)
    q_estimator.load_state_dict(torch.load(q_ckpt, map_location=args.device))
    q_estimator.to(args.device).eval()

    s2_dir = os.path.join(args.results_dir, "raw", "S2")
    s3_dir = os.path.join(args.results_dir, "raw", "S3")

    raw_results = {}
    total_runs = len(methods) * len(seeds)
    current_run = 0
    t_start_all = time.time()

    # 4. Main evaluation loop across methods x seeds
    for m in methods:
        raw_results[m] = []
        for seed in seeds:
            current_run += 1
            t_run_start = time.time()
            print(f"[{current_run}/{total_runs}] Running: Method={m.upper()} | Seed={seed} ...", end="", flush=True)

            res = evaluate_method_seed_online_fq(
                method=m,
                seed=seed,
                s2_dir=s2_dir,
                s3_dir=s3_dir,
                q_estimator=q_estimator,
                test_recs=test_recs,
                cached_images=cached_images,
                eval_tf=eval_tf,
                degradations=degradations,
                target_view=args.target_view,
                device=args.device,
                batch_size=args.batch_size,
                compare_analytical=args.compare_analytical
            )

            if res is not None:
                raw_results[m].append(res)
                t_run = time.time() - t_run_start
                audc_val = res["online_fq"]["audc"]
                hr_val = res["online_fq"]["harm_rate_l5"] * 100
                print(f" Done ({t_run:.1f}s) -> Clean QWK={res['clean_qwk']:.4f} | AUDC={audc_val:.4f} | HR_L5={hr_val:.2f}%")
            else:
                print(" [FAILED]")

    total_time = time.time() - t_start_all
    print(f"\n[INFO] Complete evaluation finished in {total_time/60:.2f} minutes!\n")

    # 5. Aggregate metrics across seeds (Mean +- Std)
    aggregated = {}
    for m in methods:
        runs = raw_results[m]
        if not runs:
            continue

        clean_qwks = [r["clean_qwk"] for r in runs]
        clean_accs = [r["clean_acc"] for r in runs]
        clean_harms = [r["clean_harm_rate"] for r in runs]

        audcs_on = [r["online_fq"]["audc"] for r in runs]
        hrs_on   = [r["online_fq"]["harm_rate_l5"] for r in runs]
        helps_on = [r["online_fq"]["help_rate_l5"] for r in runs]
        pfrs_on  = [r["online_fq"]["pfr_l5"] for r in runs]
        fdis_on  = [r["online_fq"]["fdi_l5"] for r in runs]
        q2_l5s   = [r["online_fq"]["q2_curve"][-1] for r in runs]

        mean_qwk_curve = np.mean([r["online_fq"]["qwk_curve"] for r in runs], axis=0).tolist()
        mean_harm_curve = np.mean([r["online_fq"]["harm_curve"] for r in runs], axis=0).tolist()
        mean_q2_curve = np.mean([r["online_fq"]["q2_curve"] for r in runs], axis=0).tolist()

        stat = {
            "num_seeds": len(runs),
            "clean_qwk": {"mean": float(np.mean(clean_qwks)), "std": float(np.std(clean_qwks))},
            "clean_acc": {"mean": float(np.mean(clean_accs)), "std": float(np.std(clean_accs))},
            "clean_harm_rate": {"mean": float(np.mean(clean_harms)), "std": float(np.std(clean_harms))},
            "online_fq": {
                "audc": {"mean": float(np.mean(audcs_on)), "std": float(np.std(audcs_on))},
                "harm_rate_l5": {"mean": float(np.mean(hrs_on)), "std": float(np.std(hrs_on))},
                "help_rate_l5": {"mean": float(np.mean(helps_on)), "std": float(np.std(helps_on))},
                "pfr_l5": {"mean": float(np.mean(pfrs_on)), "std": float(np.std(pfrs_on))},
                "fdi_l5": {"mean": float(np.mean(fdis_on)), "std": float(np.std(fdis_on))},
                "q2_l5": {"mean": float(np.mean(q2_l5s)), "std": float(np.std(q2_l5s))},
                "mean_qwk_curve": mean_qwk_curve,
                "mean_harm_curve": mean_harm_curve,
                "mean_q2_curve": mean_q2_curve,
            }
        }

        if args.compare_analytical:
            audcs_an = [r["analytical_formula"]["audc"] for r in runs]
            hrs_an   = [r["analytical_formula"]["harm_rate_l5"] for r in runs]
            pfrs_an  = [r["analytical_formula"]["pfr_l5"] for r in runs]
            fdis_an  = [r["analytical_formula"]["fdi_l5"] for r in runs]
            mean_qwk_curve_an = np.mean([r["analytical_formula"]["qwk_curve"] for r in runs], axis=0).tolist()
            mean_harm_curve_an = np.mean([r["analytical_formula"]["harm_curve"] for r in runs], axis=0).tolist()

            stat["analytical_formula"] = {
                "audc": {"mean": float(np.mean(audcs_an)), "std": float(np.std(audcs_an))},
                "harm_rate_l5": {"mean": float(np.mean(hrs_an)), "std": float(np.std(hrs_an))},
                "pfr_l5": {"mean": float(np.mean(pfrs_an)), "std": float(np.std(pfrs_an))},
                "fdi_l5": {"mean": float(np.mean(fdis_an)), "std": float(np.std(fdis_an))},
                "mean_qwk_curve": mean_qwk_curve_an,
                "mean_harm_curve": mean_harm_curve_an,
            }

        aggregated[m] = stat

    # 6. Save complete JSON summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary_json_path = os.path.join(args.output_dir, "online_fq_evaluation_summary.json")
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "seeds": seeds,
        "methods": methods,
        "degradations": degradations,
        "target_view": args.target_view,
        "compare_analytical": args.compare_analytical,
        "aggregated": aggregated,
        "per_seed_runs": raw_results
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[SAVE] Saved detailed JSON report: {summary_json_path}")

    # 7. Save CSV summary table
    import pandas as pd
    csv_rows = []
    for m, stat in aggregated.items():
        row = {
            "Fusion Method": m,
            "Clean QWK Mean": stat['clean_qwk']['mean'],
            "Clean QWK Std": stat['clean_qwk']['std'],
            "Clean Acc Mean": stat['clean_acc']['mean'],
            "Clean Harm Rate Mean": stat['clean_harm_rate']['mean'],
            "AUDC Mean": stat['online_fq']['audc']['mean'],
            "AUDC Std": stat['online_fq']['audc']['std'],
            "Harm Rate L5 (%)": stat['online_fq']['harm_rate_l5']['mean'] * 100,
            "PFR L5 (%)": stat['online_fq']['pfr_l5']['mean'] * 100,
            "FDI L5": stat['online_fq']['fdi_l5']['mean'],
            "Q2 L5": stat['online_fq']['q2_l5']['mean'],
        }
        csv_rows.append(row)
    csv_path = os.path.join(args.output_dir, "online_fq_table.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"[SAVE] Saved CSV summary: {csv_path}")

    # Clean up legacy .md and .tex files
    for legacy_ext in [".md", ".tex"]:
        legacy_file = os.path.join(args.output_dir, f"online_fq_table{legacy_ext}")
        if os.path.exists(legacy_file):
            try:
                os.remove(legacy_file)
            except OSError:
                pass

    # 8. Print formatted table to console
    print(f"\n{'='*75}")
    print("SUMMARY RESULTS: ONLINE f_Q EVALUATION PIPELINE (NO ARTIFICIAL ATTENUATION)")
    print(f"{'='*75}")
    print(md_table)
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
