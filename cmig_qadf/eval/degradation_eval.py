"""
cmig_qadf/eval/degradation_eval.py
==================================
Stage S4: Controlled degradation stress testing (E2, E3, E4).
Runs inference on frozen checkpoints (no re-training).
Computes QWK trajectory across severity levels 0 to 5, AUDC, Feature Drift Index (FDI),
Prediction Flip Rate (PFR), Fusion Gain, and Harm Rate.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Any

from cmig_qadf.data.adapter import DeepDRiDAdapterDataset, get_transforms, build_pair_records
from cmig_qadf.data.degradations import DEGRADATION_TYPES, degrade_pair
from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy
from cmig_qadf.eval.new_metrics import compute_audc, compute_fdi, compute_pfr, compute_harm_rate
from torch.utils.data import DataLoader


def evaluate_degradation_grid(
    model: torch.nn.Module,
    test_records: List[Dict],
    deg_types: List[str] = None,
    levels: List[int] = None,
    target_views: List[str] = None,
    image_size: int = 448,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str = None,
) -> Dict[str, Any]:
    """
    Execute grid evaluation across 7 degradation types × 5 levels × 3 views.
    """
    model.eval()
    if deg_types is None:
        deg_types = DEGRADATION_TYPES
    if levels is None:
        levels = [0, 1, 2, 3, 4, 5]
    if target_views is None:
        target_views = ["v2", "v1", "both"]

    eval_tf = get_transforms(image_size, is_train=False, use_clahe=True)

    # Pre-load test images into RAM and resize to image_size for accelerated evaluation
    cached_images = {}
    for rec in test_records:
        pid = rec["pair_id"]
        if pid not in cached_images:
            with Image.open(rec["path1"]) as im1:
                im1_rgb = im1.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            with Image.open(rec["path2"]) as im2:
                im2_rgb = im2.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            cached_images[pid] = (im1_rgb, im2_rgb)

    # 1. Clean reference baseline (Level 0)
    clean_ds = DeepDRiDAdapterDataset(test_records, transform=eval_tf, cached_images=cached_images)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)

    clean_f_cons = []
    clean_preds = []
    clean_labels = []

    with torch.no_grad():
        for b in clean_loader:
            im1 = b["img1"].to(device)
            im2 = b["img2"].to(device)
            q1 = b["q1"].to(device)
            q2 = b["q2"].to(device)
            out = model(im1, im2, q1, q2)
            clean_f_cons.append(out["f_cons"].cpu().numpy())
            clean_preds.append(out["pred"].cpu().numpy())
            clean_labels.append(b["label"].numpy())

    clean_f_cons = np.concatenate(clean_f_cons)
    clean_preds = np.concatenate(clean_preds)
    clean_labels = np.concatenate(clean_labels)
    clean_qwk = compute_qwk(clean_labels, clean_preds)

    grid_results = {
        "clean_qwk": clean_qwk,
        "clean_acc": compute_accuracy(clean_labels, clean_preds),
        "curves": {}
    }

    # 2. Iterate across degradation types and target views
    for v in target_views:
        grid_results["curves"][v] = {}
        for d_type in deg_types:
            qwk_list = [clean_qwk] # Level 0 is clean
            fdi_list = [0.0]
            pfr_list = [0.0]

            for lvl in [l for l in levels if l > 0]:
                # Dynamic degradation hook for dataset
                def deg_hook(img1, img2, q1, q2, sample_id, patient_id):
                    return degrade_pair(
                        img1, img2, q1, q2,
                        deg_type=d_type, level=lvl, target_view=v,
                        seed=42, sample_id=sample_id
                    )

                deg_ds = DeepDRiDAdapterDataset(test_records, transform=eval_tf, degradation_fn=deg_hook, cached_images=cached_images)
                deg_loader = DataLoader(deg_ds, batch_size=batch_size, shuffle=False)

                deg_f_cons = []
                deg_preds = []

                with torch.no_grad():
                    for b in deg_loader:
                        im1 = b["img1"].to(device)
                        im2 = b["img2"].to(device)
                        q1 = b["q1"].to(device)
                        q2 = b["q2"].to(device)
                        out = model(im1, im2, q1, q2)
                        deg_f_cons.append(out["f_cons"].cpu().numpy())
                        deg_preds.append(out["pred"].cpu().numpy())

                deg_f_cons = np.concatenate(deg_f_cons)
                deg_preds = np.concatenate(deg_preds)

                lvl_qwk = compute_qwk(clean_labels, deg_preds)
                lvl_fdi = compute_fdi(clean_f_cons, deg_f_cons)["mean_fdi"]
                lvl_pfr = compute_pfr(clean_preds, deg_preds)

                qwk_list.append(lvl_qwk)
                fdi_list.append(lvl_fdi)
                pfr_list.append(lvl_pfr)

            audc_val = compute_audc(qwk_list)
            grid_results["curves"][v][d_type] = {
                "qwk_curve": qwk_list,
                "fdi_curve": fdi_list,
                "pfr_curve": pfr_list,
                "audc": audc_val,
            }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(grid_results, f, indent=2)

    return grid_results
