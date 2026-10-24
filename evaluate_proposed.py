"""
evaluate_proposed.py
====================
Standalone evaluation script for the proposed QADF-Net model.
Evaluates the 5 pretrained checkpoints (seeds 42-46) on the clean DeepDRiD dual-view test cohort.
Computes per-seed metrics and 5-seed ensembled performance (QWK, Accuracy, AUC, Quadratic Kappa).

Usage:
    python evaluate_proposed.py
    python evaluate_proposed.py --device cuda --data_dir DeepDRiD
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cmig_qadf.models.dualview_model import DualViewGradingModel
from cmig_qadf.models.q_estimator import QEstimator
from cmig_qadf.data.split import get_patient_splits
from cmig_qadf.data.adapter import build_pair_records, DeepDRiDAdapterDataset, get_transforms
from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy, compute_auc, compute_per_class_kappa


def find_deepdrid_dir(base_dir: str = "DeepDRiD") -> str:
    """Locate DeepDRiD image directory containing partitions."""
    candidates = [
        os.path.join(base_dir, "regular_fundus_images"),
        base_dir,
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "regular-fundus-training")):
            return c
    return base_dir


def load_proposed_model(ckpt_path: str, q_estimator_path: str, device: str = "cpu") -> DualViewGradingModel:
    """Instantiate QADF-Net model and load pretrained weights."""
    q_estimator = None
    if os.path.exists(q_estimator_path):
        q_estimator = QEstimator(pretrained=False)
        q_estimator.load_state_dict(torch.load(q_estimator_path, map_location=device))
        q_estimator.to(device).eval()

    model = DualViewGradingModel(
        backbone_name="efficientnet_b4",
        pretrained=False,
        token_dim=128,
        grid_size=14,
        num_heads=4,
        fusion_type="qc_sca",
        qc_mode="all",
        use_sink_token=True,
        q_source="learned" if q_estimator is not None else "const1",
        q_estimator=q_estimator
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5 Pretrained Checkpoints of Proposed QADF-Net")
    parser.add_argument("--data_dir", type=str, default="DeepDRiD", help="Path to DeepDRiD dataset root")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--output_csv", type=str, default="results/tables/clean_test_evaluation_proposed.csv", help="Output CSV path")
    args = parser.parse_args()

    deepdrid_path = find_deepdrid_dir(args.data_dir)
    splits = get_patient_splits()
    test_pids = splits["test"]

    print("=" * 70)
    print("QADF-Net: Clean Dual-View Evaluation on Test Cohort (Proposed Model)")
    print(f"Device: {args.device} | Target Seeds: {args.seeds}")
    print(f"Dataset root: {deepdrid_path}")
    print("=" * 70)

    test_recs = build_pair_records(deepdrid_path, test_pids)
    if not test_recs:
        print(f"\n[WARNING] No test images found in '{deepdrid_path}'!")
        print("Please ensure DeepDRiD fundus images are placed in the dataset folder as described in README.md.")
        return

    print(f"[INFO] Loaded {len(test_recs)} dual-view image pairs for testing.")
    tf = get_transforms(image_size=448, is_train=False, use_clahe=True)
    dataset = DeepDRiDAdapterDataset(test_recs, transform=tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true = np.array([r["label"] for r in test_recs])
    q_ckpt = "results/checkpoints/q_estimator_best.pt"
    if not os.path.exists(q_ckpt):
        q_ckpt = "checkpoints/q_estimator_best.pt"

    results = []
    all_seed_probs = []

    for seed in args.seeds:
        # Check potential checkpoint paths
        ckpt_candidates = [
            f"checkpoints/qadf_net_seed{seed}.pt",
            f"results/raw/S2/e1_qc_sca_efficientnet_b4_seed{seed}_clean/best_checkpoint.pt"
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

        if ckpt_path is None:
            print(f"[WARN] Checkpoint for seed {seed} not found! Skipping.")
            continue

        print(f"\n---> Evaluating Seed {seed} from {ckpt_path} ...")
        model = load_proposed_model(ckpt_path, q_ckpt, device=args.device)

        seed_preds = []
        seed_probs = []

        with torch.no_grad():
            for batch in loader:
                img1 = batch["img1"].to(args.device)
                img2 = batch["img2"].to(args.device)
                q1 = batch["q1"].to(args.device)
                q2 = batch["q2"].to(args.device)

                out = model(img1, img2, q1, q2)
                p = out["pred"].cpu().numpy()
                pr = out["probs"].cpu().numpy()

                seed_preds.extend(p)
                seed_probs.extend(pr)

        seed_preds = np.array(seed_preds)
        seed_probs = np.array(seed_probs)
        all_seed_probs.append(seed_probs)

        qwk = compute_qwk(y_true, seed_preds)
        acc = compute_accuracy(y_true, seed_preds)
        auc = compute_auc(y_true, seed_probs)

        print(f"     Seed {seed} -> QWK: {qwk:.4f} | Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")
        results.append({
            "Seed": seed,
            "QWK": round(qwk, 4),
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4)
        })

    if not results:
        print("[ERROR] No valid seed checkpoints could be evaluated.")
        return

    # Ensemble evaluation (soft voting)
    if len(all_seed_probs) > 1:
        ens_probs = np.mean(all_seed_probs, axis=0)
        ens_preds = np.argmax(ens_probs, axis=1)
        ens_qwk = compute_qwk(y_true, ens_preds)
        ens_acc = compute_accuracy(y_true, ens_preds)
        ens_auc = compute_auc(y_true, ens_probs)

        print("-" * 70)
        print(f"5-Seed Ensemble -> QWK: {ens_qwk:.4f} | Accuracy: {ens_acc*100:.2f}% | AUC: {ens_auc:.4f}")
        results.append({
            "Seed": "Ensemble (5 Seeds)",
            "QWK": round(ens_qwk, 4),
            "Accuracy": round(ens_acc, 4),
            "AUC": round(ens_auc, 4)
        })

    # Summary statistics
    seed_qwks = [r["QWK"] for r in results if isinstance(r["Seed"], int)]
    seed_accs = [r["Accuracy"] for r in results if isinstance(r["Seed"], int)]
    seed_aucs = [r["AUC"] for r in results if isinstance(r["Seed"], int)]

    print("=" * 70)
    print(f"Summary (Mean ± Std over {len(seed_qwks)} seeds):")
    print(f"  QWK:      {np.mean(seed_qwks):.4f} ± {np.std(seed_qwks):.4f}")
    print(f"  Accuracy: {np.mean(seed_accs)*100:.2f}% ± {np.std(seed_accs)*100:.2f}%")
    print(f"  AUC:      {np.mean(seed_aucs):.4f} ± {np.std(seed_aucs):.4f}")
    print("=" * 70)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"[SAVE] Exported results to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
