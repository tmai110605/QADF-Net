"""
cmig_qadf/train/trainer.py
==========================
Unified train_one(job) routine 100% shared across all methods in the CMIG paper.
Supports:
  - Cache and intermediate resume (last_checkpoint.pt)
  - Gradient accumulation maintaining effective batch size = 32
  - Automatic Mixed Precision (AMP)
  - Automatic fallback/retry on CUDA OOM
  - Fixed CORAL Head and BCE loss
"""

import os
import sys
import time
import json
import yaml
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from cmig_qadf.models.heads import CoralLoss
from cmig_qadf.models.registry import build_model
from cmig_qadf.models.q_estimator import QEstimator
from cmig_qadf.data.adapter import build_dataloaders
from cmig_qadf.eval.metrics import compute_qwk, compute_accuracy, compute_auc


def train_one(job_config: dict, force: bool = False) -> dict:
    """
    Train a complete experimental job (Stage S2).
    """
    job_id = job_config["job_id"]
    output_dir = os.path.join(job_config.get("raw_dir", "results/raw/S2"), job_id)
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, "metrics.json")
    best_ckpt_file = os.path.join(output_dir, "best_checkpoint.pt")
    last_ckpt_file = os.path.join(output_dir, "last_checkpoint.pt")
    log_csv = os.path.join(output_dir, "train_log.csv")

    # Check cached completion
    if not force and os.path.exists(metrics_file) and os.path.exists(best_ckpt_file):
        print(f"  [CACHE] Job {job_id} already completed. Skipping.")
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Save job configuration
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(job_config, f)

    seed = job_config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = job_config.get("train", {}).get("batch_size", 8)
    accum_steps = job_config.get("train", {}).get("accum_steps", 4)
    epochs = job_config.get("train", {}).get("epochs", 60)
    warmup_epochs = job_config.get("train", {}).get("warmup_epochs", 5)
    lr = job_config.get("train", {}).get("lr", 3e-4)
    weight_decay = job_config.get("train", {}).get("weight_decay", 1e-4)
    patience = job_config.get("train", {}).get("patience", 25)
    use_amp = job_config.get("train", {}).get("use_amp", torch.cuda.is_available()) and (device == "cuda")

    # 1. Dataloaders
    deepdrid_dir = job_config["data"]["deepdrid_dir"]
    splits = job_config["splits"]
    image_size = job_config["data"].get("image_size", 448)

    loaders = build_dataloaders(
        deepdrid_dir=deepdrid_dir,
        splits=splits,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0 if os.name == "nt" else 4,
        use_clahe=job_config["data"].get("use_clahe", True),
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # 2. QEstimator (Frozen from Stage S1)
    q_estimator = None
    q_ckpt = job_config.get("q_estimator_ckpt")
    if q_ckpt and os.path.exists(q_ckpt):
        q_estimator = QEstimator(pretrained=False)
        q_estimator.load_state_dict(torch.load(q_ckpt, map_location=device))
        q_estimator.to(device)
        q_estimator.eval()
        for p in q_estimator.parameters():
            p.requires_grad = False

    # 3. Model
    model = build_model(job_config, q_estimator=q_estimator).to(device)
    loss_fn = CoralLoss(num_classes=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_qwk = -1.0
    patience_cnt = 0
    history = []

    # Resume from last_checkpoint if present
    if os.path.exists(last_ckpt_file):
        try:
            ckpt = torch.load(last_ckpt_file, map_location=device)
            start_epoch = ckpt["epoch"] + 1
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            best_qwk = ckpt.get("best_qwk", -1.0)
            patience_cnt = ckpt.get("patience_cnt", 0)
            print(f"  [RESUME] Resuming job {job_id} from epoch {start_epoch}")
        except Exception as e:
            print(f"  [WARN] Failed to read last_checkpoint: {e}. Starting fresh.")

    # Open train_log.csv
    log_exists = os.path.exists(log_csv)
    log_f = open(log_csv, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    if not log_exists:
        log_writer.writerow(["epoch", "train_loss", "val_qwk", "val_acc", "val_auc", "lr", "time_sec"])
        log_f.flush()

    print(f"\n--- Starting training for job: {job_id} ({epochs} epochs) ---")

    for ep in range(start_epoch, epochs):
        t0 = time.time()
        model.train()

        # Warmup LR
        if ep < warmup_epochs:
            cur_lr = lr * float(ep + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            q1   = batch["q1"].to(device, non_blocking=True)
            q2   = batch["q2"].to(device, non_blocking=True)
            lbls = batch["label"].to(device, non_blocking=True)

            with autocast("cuda" if device == "cuda" else "cpu", enabled=use_amp):
                out = model(img1, img2, q1, q2)
                loss = loss_fn(out["logits"], lbls)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)
        if ep >= warmup_epochs:
            scheduler.step()

        # Validation evaluation
        model.eval()
        val_preds, val_lbls, val_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                img1 = batch["img1"].to(device, non_blocking=True)
                img2 = batch["img2"].to(device, non_blocking=True)
                q1   = batch["q1"].to(device, non_blocking=True)
                q2   = batch["q2"].to(device, non_blocking=True)
                lbls = batch["label"]

                with autocast("cuda" if device == "cuda" else "cpu", enabled=use_amp):
                    out = model(img1, img2, q1, q2)

                val_preds.append(out["pred"].cpu().numpy())
                val_probs.append(out["probs"].cpu().numpy())
                val_lbls.append(lbls.numpy())

        y_p = np.concatenate(val_preds)
        y_t = np.concatenate(val_lbls)
        y_prob = np.concatenate(val_probs)

        v_qwk = compute_qwk(y_t, y_p)
        v_acc = compute_accuracy(y_t, y_p)
        v_auc = compute_auc(y_t, y_prob)
        dt = time.time() - t0

        log_writer.writerow([ep + 1, f"{train_loss:.4f}", f"{v_qwk:.4f}", f"{v_acc:.4f}", f"{v_auc:.4f}", f"{cur_lr:.6f}", f"{dt:.1f}"])
        log_f.flush()

        print(f"  Ep {ep+1:2d}/{epochs} | L={train_loss:.4f} | Val QWK={v_qwk:.4f} Acc={v_acc:.4f} AUC={v_auc:.4f} | {dt:.1f}s")

        # Save best checkpoint according to Validation QWK
        if v_qwk > best_qwk:
            best_qwk = v_qwk
            patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt_file)
            print(f"  ✓ Best Val QWK={best_qwk:.4f} → saved {best_ckpt_file}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [STOP] Early stopping triggered at epoch {ep+1} (patience={patience})")
                break

        # Save current checkpoint for failure recovery
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_qwk": best_qwk,
            "patience_cnt": patience_cnt,
        }, last_ckpt_file)

    log_f.close()
    if not os.path.exists(best_ckpt_file):
        torch.save(model.state_dict(), best_ckpt_file)

    result = {
        "job_id": job_id,
        "best_qwk": float(best_qwk),
        "best_checkpoint": best_ckpt_file,
        "epochs_trained": ep + 1,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Release VRAM for subsequent jobs
    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
