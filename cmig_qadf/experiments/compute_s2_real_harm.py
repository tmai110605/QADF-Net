"""
cmig_qadf/experiments/compute_s2_real_harm.py
=============================================
Compute empirical Harm Rate, Help Rate, Tie Rate, PFR, and QWK
on the test cohort across 6 degradation severity levels (Levels 0 to 5) across 5 seeds.

Supports all fusion baselines (excluding q_weighted):
  - qc_sca (Proposed architecture)
  - concat (Classic concatenation)
  - average (Algebraic mean)
  - co_attn (Dual-view Transformer co-attention)
  - cross_attn (Standard unidirectional cross-attention)
  - max (Element-wise maximum)

Automatically caches and stores results to results/processed/s2_harm_empirical.json.
"""

import os
import sys
import json
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))

from cmig_qadf.models.registry import build_model
from cmig_qadf.models.q_estimator import QEstimator
from cmig_qadf.data.adapter import build_pair_records, DeepDRiDAdapterDataset, get_transforms
from cmig_qadf.data.degradations import degrade_pair, DEGRADATION_TYPES
from cmig_qadf.eval.metrics import compute_qwk
from cmig_qadf.eval.new_metrics import compute_harm_rate, compute_pfr


# All evaluated fusion methods (strictly excludes q_weighted per protocol)
ALL_FUSION_METHODS = ["qc_sca", "concat", "average", "co_attn", "cross_attn", "max"]


def find_deepdrid_dir(base_dir: str = "DeepDRiD") -> str:
    """Automatically discover directory containing DeepDRiD dataset partitions."""
    candidates = [
        os.path.join(base_dir, "regular_fundus_images"),
        base_dir,
        "DeepDRiD/regular_fundus_images",
        "DeepDRiD",
        "/home/thaimq/DiabeticRetinopathy/DeepDRiD/regular_fundus_images",
        "/home/thaimq/DiabeticRetinopathy/DeepDRiD",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "regular-fundus-training")) or \
           os.path.exists(os.path.join(c, "regular-fundus-validation")) or \
           os.path.exists(os.path.join(c, "Online-Challenge1&2-Evaluation")):
            return c
    return os.path.join(base_dir, "regular_fundus_images")


def run_empirical_harm_evaluation(
    results_dir: str = "results",
    deepdrid_dir: str = None,
    target_view: str = "v2",
    methods: list = None,
    seeds: list = None,
    degradations: list = None,
    device: str = None,
    force: bool = False
):
    if methods is None:
        methods = ALL_FUSION_METHODS
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if degradations is None:
        degradations = DEGRADATION_TYPES
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Filter out q_weighted if passed accidentally
    methods = [m for m in methods if m != "q_weighted"]

    s2_dir = os.path.join(results_dir, "raw", "S2")
    s3_dir = os.path.join(results_dir, "raw", "S3")
    split_manifest = os.path.join(results_dir, "manifests", "split_manifest.json")
    out_json = os.path.join(results_dir, "processed", "s2_harm_empirical.json")

    # 1. Check checkpoints folder
    if not os.path.exists(s2_dir):
        print(f"[ERROR] Checkpoints directory not found: {s2_dir}")
        print("  -> Please run this script with available checkpoints under results/raw/S2/!")
        return None

    if not os.path.exists(split_manifest):
        print(f"[ERROR] Split manifest not found: {split_manifest}")
        return None

    with open(split_manifest, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # 2. Automatically locate DeepDRiD directory
    if deepdrid_dir is None:
        deepdrid_dir = find_deepdrid_dir("DeepDRiD")
    print(f"[INFO] Using dataset directory: {deepdrid_dir}")

    test_recs = build_pair_records(deepdrid_dir, splits["test"])
    if len(test_recs) == 0:
        alt = os.path.join(deepdrid_dir, "regular_fundus_images")
        if os.path.exists(alt):
            test_recs = build_pair_records(alt, splits["test"])
            deepdrid_dir = alt

    test_recs = sorted(test_recs, key=lambda r: r["pair_id"])
    print(f"[INFO] Evaluating empirical Harm Rate on {len(test_recs)} test samples ({device})...")
    print(f"[INFO] Evaluation methods: {methods}")

    if len(test_recs) == 0:
        print("[ERROR] No test samples found! Please check DeepDRiD image directory paths.")
        return None

    # 3. Pre-cache test images into RAM (~120MB)
    eval_tf = get_transforms(448, is_train=False, use_clahe=True)
    cached_images = {}
    print("  -> Pre-caching test images into memory...")
    for rec in test_recs:
        pid = rec["pair_id"]
        if pid not in cached_images:
            try:
                with Image.open(rec["path1"]) as im1:
                    im1_rgb = im1.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
                with Image.open(rec["path2"]) as im2:
                    im2_rgb = im2.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
                cached_images[pid] = (im1_rgb, im2_rgb)
            except Exception as e:
                print(f"  [WARN] Failed to read image {pid}: {e}")

    # 4. Load Q-Estimator if available
    q_ckpt = os.path.join(results_dir, "checkpoints", "q_estimator_best.pt")
    q_estimator = None
    if os.path.exists(q_ckpt):
        q_estimator = QEstimator(pretrained=False)
        q_estimator.load_state_dict(torch.load(q_ckpt, map_location=device))
        q_estimator.to(device).eval()

    all_methods_summary = {}
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                raw_j = json.load(f)
            if "0" in raw_j:
                all_methods_summary["qc_sca"] = raw_j
            else:
                all_methods_summary = raw_j
        except Exception:
            pass

    # 5. Iterate across fusion methods
    for m in methods:
        if not force and m in all_methods_summary and ("5" in all_methods_summary[m] or 5 in all_methods_summary[m]):
            print(f"\n[SKIP] Method '{m}' already exists in {out_json}. Skipping.")
            continue

        print(f"\n{'#'*65}\n### EMPIRICAL EVALUATION: {m.upper()} (5 SEEDS x 6 LEVELS) ###\n{'#'*65}")
        m_results = {lvl: {"harm_rates": [], "help_rates": [], "tie_rates": [], "pfrs": [], "qwks": []} for lvl in range(6)}

        for seed in seeds:
            job_name = f"e1_{m}_efficientnet_b4_seed{seed}_clean"
            ckpt_path = os.path.join(s2_dir, job_name, "best_checkpoint.pt")
            cfg_path  = os.path.join(s2_dir, job_name, "config.yaml")
            v1_p = os.path.join(s3_dir, f"e1_single_v1_efficientnet_b4_seed{seed}_clean.parquet")
            v2_p = os.path.join(s3_dir, f"e1_single_v2_efficientnet_b4_seed{seed}_clean.parquet")
            m_p  = os.path.join(s3_dir, f"{job_name}.parquet")
            qc_p = os.path.join(s3_dir, f"e1_qc_sca_efficientnet_b4_seed{seed}_clean.parquet")

            if not os.path.exists(ckpt_path):
                print(f"  [WARN] Skipping seed {seed} for {m} due to missing checkpoint: {ckpt_path}")
                continue

            print(f"\n--- Method: {m} | Seed: {seed} ---")
            df_v1 = pd.read_parquet(v1_p).sort_values("sample_id").reset_index(drop=True)
            df_v2 = pd.read_parquet(v2_p).sort_values("sample_id").reset_index(drop=True)
            df_m  = pd.read_parquet(m_p).sort_values("sample_id").reset_index(drop=True)
            df_qc = pd.read_parquet(qc_p).sort_values("sample_id").reset_index(drop=True)

            clean_labels = df_m["label"].values
            clean_preds  = df_m["pred"].values
            
            # Determine best single view objectively per sample by effective quality score
            pred_best = np.where(df_qc["q1_effective"] >= df_qc["q2_effective"], df_v1["pred"].values, df_v2["pred"].values)

            # Level 0 (Clean)
            h0 = compute_harm_rate(clean_labels, clean_preds, pred_best)
            qwk0 = compute_qwk(clean_labels, clean_preds)
            m_results[0]["harm_rates"].append(h0["harm_rate"])
            m_results[0]["help_rates"].append(h0["help_rate"])
            m_results[0]["tie_rates"].append(h0["tie_rate"])
            m_results[0]["pfrs"].append(0.0)
            m_results[0]["qwks"].append(qwk0)
            print(f"  [Level 0 - Clean]: Harm = {h0['harm_rate']*100:.2f}%, Help = {h0['help_rate']*100:.2f}%, QWK = {qwk0:.4f}")

            # Load model
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            model = build_model(cfg, q_estimator=q_estimator if m == "qc_sca" else None)
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
            model.to(device).eval()

            # Levels 1 to 5
            for lvl in range(1, 6):
                deg_harms, deg_helps, deg_ties, deg_pfrs, deg_qwks = [], [], [], [], []
                for deg in degradations:
                    def deg_hook(img1, img2, q1, q2, sample_id, patient_id):
                        return degrade_pair(img1, img2, q1, q2, deg_type=deg, level=lvl, target_view=target_view, seed=seed, sample_id=sample_id)

                    deg_ds = DeepDRiDAdapterDataset(test_recs, transform=eval_tf, degradation_fn=deg_hook, cached_images=cached_images)
                    deg_loader = DataLoader(deg_ds, batch_size=16, shuffle=False)

                    deg_preds = []
                    with torch.no_grad():
                        for b in deg_loader:
                            im1 = b["img1"].to(device)
                            im2 = b["img2"].to(device)
                            q1 = b["q1"].to(device)
                            q2 = b["q2"].to(device)
                            out = model(im1, im2, q1, q2)
                            deg_preds.append(out["pred"].cpu().numpy())

                    if len(deg_preds) == 0:
                        continue
                    deg_preds = np.concatenate(deg_preds)

                    hl = compute_harm_rate(clean_labels, deg_preds, pred_best)
                    pfr = compute_pfr(clean_preds, deg_preds)
                    qwk = compute_qwk(clean_labels, deg_preds)

                    deg_harms.append(hl["harm_rate"])
                    deg_helps.append(hl["help_rate"])
                    deg_ties.append(hl["tie_rate"])
                    deg_pfrs.append(pfr)
                    deg_qwks.append(qwk)

                m_harm_lvl = float(np.mean(deg_harms))
                m_help_lvl = float(np.mean(deg_helps))
                m_tie_lvl  = float(np.mean(deg_ties))
                m_pfr_lvl  = float(np.mean(deg_pfrs))
                m_qwk_lvl  = float(np.mean(deg_qwks))

                m_results[lvl]["harm_rates"].append(m_harm_lvl)
                m_results[lvl]["help_rates"].append(m_help_lvl)
                m_results[lvl]["tie_rates"].append(m_tie_lvl)
                m_results[lvl]["pfrs"].append(m_pfr_lvl)
                m_results[lvl]["qwks"].append(m_qwk_lvl)

                print(f"  [Level {lvl}]: Harm = {m_harm_lvl*100:.2f}%, Help = {m_help_lvl*100:.2f}%, PFR = {m_pfr_lvl*100:.2f}%, QWK = {m_qwk_lvl:.4f}")

        # Aggregate across seeds for method m
        m_summary = {}
        print(f"\n--- Aggregating 5 seeds for {m.upper()} ---")
        for lvl in range(6):
            if not m_results[lvl]["harm_rates"]:
                continue
            m_summary[str(lvl)] = {
                "harm_rate_mean": float(np.mean(m_results[lvl]["harm_rates"])),
                "harm_rate_std": float(np.std(m_results[lvl]["harm_rates"])),
                "help_rate_mean": float(np.mean(m_results[lvl]["help_rates"])),
                "tie_rate_mean": float(np.mean(m_results[lvl]["tie_rates"])),
                "pfr_mean": float(np.mean(m_results[lvl]["pfrs"])),
                "pfr_std": float(np.std(m_results[lvl]["pfrs"])),
                "qwk_mean": float(np.mean(m_results[lvl]["qwks"])),
                "qwk_std": float(np.std(m_results[lvl]["qwks"])),
            }
            print(f"  Level {lvl}: Harm Rate = {m_summary[str(lvl)]['harm_rate_mean']*100:.2f}% | PFR = {m_summary[str(lvl)]['pfr_mean']*100:.2f}% | QWK = {m_summary[str(lvl)]['qwk_mean']:.4f}")

        all_methods_summary[m] = m_summary

        # Save immediately per method to safeguard against interruption
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_methods_summary, f, indent=2)
        print(f"  -> Updated results for {m} in {out_json}")

    print("\n================ EMPIRICAL SUMMARY ACROSS METHODS (CMIG S2) ================")
    print(f"{'Method':14s} | {'Level 0 Harm':12s} | {'Level 5 Harm':12s} | {'Level 5 PFR':11s} | {'Level 5 QWK':11s}")
    print("-" * 75)
    for m, s in all_methods_summary.items():
        k0 = "0" if "0" in s else (0 if 0 in s else None)
        k5 = "5" if "5" in s else (5 if 5 in s else None)
        if k0 is not None and k5 is not None:
            h0 = s[k0]["harm_rate_mean"] * 100
            h5 = s[k5]["harm_rate_mean"] * 100
            p5 = s[k5]["pfr_mean"] * 100
            q5 = s[k5]["qwk_mean"]
            print(f"{m:14s} | {h0:11.2f}% | {h5:11.2f}% | {p5:10.2f}% | {q5:11.4f}")

    return all_methods_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate empirical harm rate and PFR across severity levels for all fusion baselines.")
    parser.add_argument("--methods", nargs="+", default=ALL_FUSION_METHODS, help="Methods to evaluate: qc_sca concat average co_attn cross_attn max (strictly excludes q_weighted)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--view", default="v2", help="Target view: v2, v1, both")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if method exists in json")
    args = parser.parse_args()

    run_empirical_harm_evaluation(methods=args.methods, seeds=args.seeds, target_view=args.view, force=args.force)
