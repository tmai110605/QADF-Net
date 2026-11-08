"""
qadfnet/experiments/eval_mfiddr.py
====================================
Comprehensive External Cohort Evaluation on MFIDDR (Multi-Field Images for Diabetic Retinopathy).
Reference: Haoran Li et al., arXiv:2602.21944 (2026).

Clinical Pairing Protocol for Dual-View CMIG Evaluation:
  - View 1: id1 (Field 1: Central / Macula-centred, corresponds to DeepDRiD View 1)
  - View 2: id2 (Field 2: Nasal / Optic-disc centred, corresponds to DeepDRiD View 2)
  - Real fundus pairs evaluated under zero-shot transfer across distinct camera optics and clinical centers.

Experimental Protocols:
  - Clean Cross-Cohort Evaluation (Zero-shot generalization across domain shift)
  - Controlled Asymmetric Optical Stress-Test (Levels L0 to L5 applied to View 2)
  - Online f_Q quality estimation from pixels (MobileNetV3) on GPU runtime
  - Complete metrics: QWK, Accuracy, AUC, AUDC, Harm Rate, Delta Harm, PFR, FDI
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

from qadfnet.models.dualview_model import DualViewGradingModel
from qadfnet.models.q_estimator import QEstimator
from qadfnet.models.registry import build_model
from qadfnet.data.adapter import get_transforms
from qadfnet.data.mfiddr import build_mfiddr_records, pre_cache_mfiddr_images, MFIDDRDataset
from qadfnet.eval.metrics import compute_qwk, compute_accuracy, compute_auc
from qadfnet.eval.new_metrics import compute_audc, compute_harm_rate, compute_pfr, compute_fdi


def load_model_checkpoint(
    method: str,
    seed: int,
    s2_dir: str = "results/raw/S2",
    device: str = "cpu"
) -> DualViewGradingModel:
    """Load model checkpoint for a specific method and seed."""
    job_name = f"e1_{method}_efficientnet_b4_seed{seed}_clean"
    job_dir = os.path.join(s2_dir, job_name)
    cfg_path = os.path.join(job_dir, "config.yaml")
    ckpt_path = os.path.join(job_dir, "best_checkpoint.pt")

    # Fallback for standalone checkpoints
    if not os.path.exists(ckpt_path):
        if method == "qc_sca":
            alt_ckpt = f"checkpoints/qadf_net_seed{seed}.pt"
            if os.path.exists(alt_ckpt):
                ckpt_path = alt_ckpt

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for method '{method}' seed {seed} (checked {ckpt_path})")

    # Load config if available, else construct default
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg)
    else:
        # Default architecture fallback
        fusion_type = method
        qc_mode = "all"
        use_sink = True
        if method == "qc_sca":
            fusion_type = "qc_sca"
        elif method == "cross_attn":
            fusion_type = "cross_attn"
            qc_mode = "none"
            use_sink = False
        elif method == "q_weighted":
            fusion_type = "q_weighted"
        elif method == "concat":
            fusion_type = "concat"
        elif method == "average":
            fusion_type = "average"
        elif method == "single_v1":
            fusion_type = "single_v1"

        model = DualViewGradingModel(
            backbone_name="efficientnet_b4",
            pretrained=False,
            token_dim=128,
            grid_size=14,
            num_heads=4,
            dropout=0.1,
            fusion_type=fusion_type,
            qc_mode=qc_mode,
            use_sink_token=use_sink,
            num_classes=5
        )

    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def run_inference(
    model: nn.Module,
    q_estimator: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference with dynamic online f_Q quality evaluation."""
    preds_list = []
    probs_list = []
    f_cons_list = []
    q1_list = []
    q2_list = []
    labels_list = []

    model.eval()
    q_estimator.eval()

    with torch.no_grad():
        for batch in dataloader:
            im1 = batch["img1"].to(device, non_blocking=True)
            im2 = batch["img2"].to(device, non_blocking=True)
            lbl = batch["label"].numpy()

            # Dynamic online quality estimation at 224x224
            im1_224 = F.interpolate(im1, size=(224, 224), mode="bilinear", align_corners=False)
            im2_224 = F.interpolate(im2, size=(224, 224), mode="bilinear", align_corners=False)

            q1_hat = q_estimator(im1_224)
            q2_hat = q_estimator(im2_224)

            out = model(im1, im2, q1_hat, q2_hat)

            preds_list.append(out["pred"].cpu().numpy())
            probs_list.append(out["probs"].cpu().numpy())
            f_cons_list.append(out["f_cons"].cpu().numpy())
            q1_list.append(q1_hat.cpu().squeeze().numpy())
            q2_list.append(q2_hat.cpu().squeeze().numpy())
            labels_list.append(lbl)

    preds = np.concatenate(preds_list)
    probs = np.concatenate(probs_list, axis=0)
    f_cons = np.concatenate(f_cons_list, axis=0)
    q1 = np.concatenate([np.atleast_1d(x) for x in q1_list])
    q2 = np.concatenate([np.atleast_1d(x) for x in q2_list])
    labels = np.concatenate(labels_list)

    return preds, probs, f_cons, q1, q2, labels


def evaluate_mfiddr_benchmark(
    mfiddr_dir: str = "MFIDDR",
    split: str = "test",
    max_samples: int = 0,
    methods: List[str] = ["qc_sca", "q_weighted", "cross_attn", "concat", "average", "single_v1"],
    seeds: List[int] = [42, 43, 44, 45, 46],
    deg_types: List[str] = ["defocus", "gaussian_blur", "dark", "glare"],
    deg_levels: List[int] = [0, 1, 2, 3, 4, 5],
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "results/processed"
) -> Dict[str, Any]:
    """Execute full external evaluation on the MFIDDR cohort."""
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  CMIG BENCHMARK: EXTERNAL EVALUATION ON MFIDDR COHORT")
    print(f"  Dataset Directory : {mfiddr_dir} (Split: {split})")
    print(f"  Methods Evaluated : {methods}")
    print(f"  Random Seeds      : {seeds}")
    print(f"  Degradation Types : {deg_types} (Levels 0-5)")
    print(f"  Device / Batch    : {device.upper()} | Batch Size: {batch_size}")
    print(f"{'='*75}\n", flush=True)

    # 1. Parse records
    all_records = build_mfiddr_records(mfiddr_dir, split=split)
    if max_samples > 0 and max_samples < len(all_records):
        np.random.seed(42)
        indices = np.random.choice(len(all_records), max_samples, replace=False)
        records = [all_records[i] for i in indices]
        print(f"[Sampling] Selected {len(records)}/{len(all_records)} eye pairs for evaluation.", flush=True)
    else:
        records = all_records
        print(f"[Dataset] Evaluating ALL {len(records)} test eye pairs ({len(records)*2} images).", flush=True)

    # 2. Pre-cache images in RAM (448x448)
    image_cache = pre_cache_mfiddr_images(records, target_size=448)
    eval_tf = get_transforms(image_size=448, is_train=False, use_clahe=True)

    # 3. Load frozen quality estimator f_Q
    q_ckpt = "checkpoints/q_estimator_best.pt"
    if not os.path.exists(q_ckpt):
        q_ckpt = "results/checkpoints/q_estimator_best.pt"
    q_estimator = QEstimator(pretrained=False)
    q_estimator.load_state_dict(torch.load(q_ckpt, map_location=device))
    q_estimator.to(device).eval()

    # 4. Pre-compute Single-View V1 clean reference predictions (for Harm Rate baseline)
    print("\n[Step 1] Pre-computing Single-View V1 Clean Reference Predictions...", flush=True)
    clean_ds = MFIDDRDataset(records, image_cache=image_cache, transform=eval_tf, deg_level=0)
    clean_loader = DataLoader(
        clean_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )

    v1_clean_preds_per_seed = {}
    for seed in seeds:
        v1_model = load_model_checkpoint("single_v1", seed, device=device)
        p, _, _, _, _, y_true = run_inference(v1_model, q_estimator, clean_loader, device=device)
        v1_clean_preds_per_seed[seed] = p
        qwk_v1 = compute_qwk(y_true, p)
        acc_v1 = compute_accuracy(y_true, p)
        print(f"  Single-View V1 (Seed {seed}) -> QWK: {qwk_v1:.4f} | Acc: {acc_v1*100:.2f}%", flush=True)
        del v1_model

    # 5. Clean Zero-Shot Evaluation (Zero-shot generalization across domain shift)
    print("\n[Step 2] Clean Cross-Cohort Evaluation (L0 Zero-Shot Generalization)...", flush=True)
    clean_results = {}
    clean_f_cons = {}

    for method in methods:
        qwk_list, acc_list, auc_list = [], [], []
        t0_m = time.time()

        for seed in seeds:
            model = load_model_checkpoint(method, seed, device=device)
            preds, probs, f_cons, q1_s, q2_s, y_t = run_inference(model, q_estimator, clean_loader, device=device)
            
            qwk_list.append(compute_qwk(y_t, preds))
            acc_list.append(compute_accuracy(y_t, preds))
            auc_list.append(compute_auc(y_t, probs))

            if seed == seeds[0]:
                clean_f_cons[method] = f_cons

            del model

        dt_m = time.time() - t0_m
        clean_results[method] = {
            "qwk_mean": float(np.mean(qwk_list)),
            "qwk_std": float(np.std(qwk_list)),
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list)),
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list)),
            "qwk_per_seed": [float(x) for x in qwk_list],
            "acc_per_seed": [float(x) for x in acc_list],
            "auc_per_seed": [float(x) for x in auc_list],
        }
        cr = clean_results[method]
        print(f"  [{method:12s}] QWK: {cr['qwk_mean']:.4f}±{cr['qwk_std']:.4f} | Acc: {cr['acc_mean']*100:.2f}% | AUC: {cr['auc_mean']:.4f} ({dt_m:.1f}s)", flush=True)

    # 6. Asymmetric Optical Stress-Test (Levels L0 to L5 on View 2)
    print("\n[Step 3] Asymmetric Optical Stress-Test (View 2 Degraded, View 1 Clean)...", flush=True)
    stress_results = {m: {} for m in methods}

    for d_type in deg_types:
        print(f"\n--- Degradation: {d_type.upper()} ---", flush=True)
        for method in methods:
            t0_s = time.time()
            method_audc_list = []
            method_harm_l5_list = []
            method_delta_harm_list = []
            method_pfr_l5_list = []
            method_fdi_l5_list = []

            for seed in seeds:
                model = load_model_checkpoint(method, seed, device=device)
                v1_clean_preds = v1_clean_preds_per_seed[seed]
                qwk_levels = []
                harm_levels = []
                pfr_levels = []
                fdi_levels = []

                preds_l0 = None
                f_cons_l0 = None

                for level in deg_levels:
                    deg_ds = MFIDDRDataset(
                        records,
                        image_cache=image_cache,
                        transform=eval_tf,
                        deg_type=d_type,
                        deg_level=level,
                        target_view="v2",
                        seed=seed
                    )
                    deg_loader = DataLoader(
                        deg_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device == "cuda")
                    )

                    preds, probs, f_cons, _, _, y_t = run_inference(model, q_estimator, deg_loader, device=device)

                    if level == 0:
                        preds_l0 = preds
                        f_cons_l0 = f_cons

                    # Metrics
                    qwk_l = compute_qwk(y_t, preds)
                    harm_info = compute_harm_rate(y_t, preds, v1_clean_preds)
                    pfr_l = compute_pfr(preds_l0, preds)
                    fdi_info = compute_fdi(f_cons_l0, f_cons)

                    qwk_levels.append(qwk_l)
                    harm_levels.append(harm_info["harm_rate"])
                    pfr_levels.append(pfr_l)
                    fdi_levels.append(fdi_info["mean_fdi"])

                audc = compute_audc(qwk_levels)
                harm_l5 = harm_levels[-1]
                delta_harm = harm_levels[-1] - harm_levels[0]
                pfr_l5 = pfr_levels[-1]
                fdi_l5 = fdi_levels[-1]

                method_audc_list.append(audc)
                method_harm_l5_list.append(harm_l5)
                method_delta_harm_list.append(delta_harm)
                method_pfr_l5_list.append(pfr_l5)
                method_fdi_l5_list.append(fdi_l5)

                del model

            audc_mean = float(np.mean(method_audc_list))
            harm_l5_mean = float(np.mean(method_harm_l5_list))
            delta_harm_mean = float(np.mean(method_delta_harm_list))
            pfr_l5_mean = float(np.mean(method_pfr_l5_list))
            fdi_l5_mean = float(np.mean(method_fdi_l5_list))

            stress_results[method][d_type] = {
                "audc_mean": audc_mean,
                "harm_l5_mean": harm_l5_mean,
                "delta_harm_mean": delta_harm_mean,
                "pfr_l5_mean": pfr_l5_mean,
                "fdi_l5_mean": fdi_l5_mean,
                "audc_per_seed": [float(x) for x in method_audc_list],
                "harm_l5_per_seed": [float(x) for x in method_harm_l5_list],
                "delta_harm_per_seed": [float(x) for x in method_delta_harm_list],
            }

            dt_s = time.time() - t0_s
            sign = "+" if delta_harm_mean >= 0 else ""
            print(f"  [{method:12s} | {d_type:13s}] AUDC: {audc_mean:.4f} | Harm L5: {harm_l5_mean*100:.2f}% (Δ: {sign}{delta_harm_mean*100:.2f}%) | PFR L5: {pfr_l5_mean*100:.2f}% ({dt_s:.1f}s)", flush=True)

    # 7. Compile Final Output Summary
    total_time = time.time() - t_start
    summary = {
        "dataset": "MFIDDR",
        "split": split,
        "n_pairs": len(records),
        "methods": methods,
        "seeds": seeds,
        "deg_types": deg_types,
        "clean_metrics": clean_results,
        "stress_test": stress_results,
        "execution_time_sec": total_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save JSON summary
    summary_path = os.path.join(output_dir, "mfiddr_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved JSON] -> {summary_path}", flush=True)

    # 8. Build Comprehensive Markdown / CSV Summary Table
    csv_rows = []
    for method in methods:
        c_m = clean_results.get(method, {})
        audc_vals = [stress_results[method][d]["audc_mean"] for d in deg_types if d in stress_results.get(method, {})]
        harm_l5_vals = [stress_results[method][d]["harm_l5_mean"] for d in deg_types if d in stress_results.get(method, {})]
        delta_harm_vals = [stress_results[method][d]["delta_harm_mean"] for d in deg_types if d in stress_results.get(method, {})]
        pfr_l5_vals = [stress_results[method][d]["pfr_l5_mean"] for d in deg_types if d in stress_results.get(method, {})]

        sign = "+" if np.mean(delta_harm_vals) >= 0 else ""
        csv_rows.append({
            "Dataset": "MFIDDR",
            "Method": method,
            "Clean_QWK": f"{c_m.get('qwk_mean', 0.0):.4f} ± {c_m.get('qwk_std', 0.0):.4f}",
            "Clean_Accuracy": f"{c_m.get('acc_mean', 0.0)*100:.2f}%",
            "Clean_AUC": f"{c_m.get('auc_mean', 0.0):.4f}",
            "Mean_AUDC": f"{np.mean(audc_vals):.4f}" if audc_vals else "—",
            "Harm_Rate_L5": f"{np.mean(harm_l5_vals)*100:.2f}%" if harm_l5_vals else "—",
            "Delta_Harm": f"{sign}{np.mean(delta_harm_vals)*100:.2f}%" if delta_harm_vals else "—",
            "PFR_L5": f"{np.mean(pfr_l5_vals)*100:.2f}%" if pfr_l5_vals else "—",
        })

    df_out = pd.DataFrame(csv_rows)
    csv_path = "results/tables/TableS6_External_MFIDDR.csv"
    df_out.to_csv(csv_path, index=False)
    df_out.to_csv("results/tables/mfiddr_benchmark.csv", index=False)
    print(f"[Saved Table] -> {csv_path}\n", flush=True)
    print(df_out.to_string(index=False), flush=True)

    print(f"\nBenchmark completed successfully in {total_time/60:.1f} minutes.\n", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Proposed QADF-Net and Baselines on External MFIDDR Cohort")
    parser.add_argument("--mfiddr_dir", type=str, default="MFIDDR", help="Path to MFIDDR directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Split to evaluate")
    parser.add_argument("--max_samples", type=int, default=0, help="Max eye pairs to evaluate (0 = all pairs)")
    parser.add_argument("--methods", nargs="+", default=["qc_sca", "q_weighted", "cross_attn", "concat", "average", "single_v1"], help="Methods to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds (e.g. 42 or 42 43 44 45 46)")
    parser.add_argument("--deg_types", nargs="+", default=["defocus", "gaussian_blur", "dark", "glare"], help="Degradation types to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4 if sys.platform != "win32" else 0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    args = parser.parse_args()

    evaluate_mfiddr_benchmark(
        mfiddr_dir=args.mfiddr_dir,
        split=args.split,
        max_samples=args.max_samples,
        methods=args.methods,
        seeds=args.seeds,
        deg_types=args.deg_types,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )

