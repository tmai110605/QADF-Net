"""
cmig_qadf/train/train_qhat.py
=============================
Stage S1: Train image quality estimator q_hat (QEstimator).
Supervision:
  1. Regression on human expert quality ratings of clean DeepDRiD fundus photographs.
  2. Monotonic regression on randomly degraded synthetic images.
Upon completion, the best checkpoint is FROZEN and used downstream across the pipeline.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cmig_qadf.models.q_estimator import QEstimator
from cmig_qadf.data.adapter import build_pair_records
from cmig_qadf.data.degradations import DEGRADATION_TYPES, apply_single_degradation
from cmig_qadf.data.degradation_calibration import calibrate_and_verify


class QHatTrainingDataset(Dataset):
    """Dataset yielding clean and synthetically degraded images to train q_estimator."""
    def __init__(self, records, input_size: int = 224, is_train: bool = True):
        self.records = records
        self.input_size = input_size
        self.is_train = is_train

        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        # Alternate sampling between View 1 and View 2
        path = rec["path1"] if (idx % 2 == 0) else rec["path2"]
        q_clean = float(rec["q1"]) if (idx % 2 == 0) else float(rec["q2"])
        
        img_clean = Image.open(path).convert("RGB")

        if self.is_train and random.random() < 0.6:
            # Generate continuous degradation sample
            deg_type = random.choice(DEGRADATION_TYPES)
            level = random.randint(1, 5)
            s = float(level) / 5.0
            deg_img = apply_single_degradation(img_clean, deg_type, level, seed=random.randint(0, 10000), sample_id=rec["pair_id"])
            q_target = max(0.001, q_clean * (1.0 - 0.9 * s))
            tensor = self.tf(deg_img)
        else:
            tensor = self.tf(img_clean)
            q_target = q_clean

        return tensor, torch.tensor([q_target], dtype=torch.float32)


def train_q_estimator(
    deepdrid_dir: str,
    splits: dict,
    epochs: int = 15,
    lr: float = 5e-4,
    batch_size: int = 16,
    input_size: int = 224,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "results/checkpoints",
    output_dir: str = "results/processed"
) -> str:
    """
    Training routine for QEstimator. Returns the path to the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "q_estimator_best.pt")

    if os.path.exists(ckpt_path):
        print(f"  [CACHE] QEstimator checkpoint already exists: {ckpt_path}. Skipping training.")
        return ckpt_path

    print(f"\n{'='*60}")
    print(f"  [S1] Training QEstimator | Epochs={epochs} | Device={device}")
    print(f"{'='*60}")

    train_recs = build_pair_records(deepdrid_dir, splits["train"])
    val_recs   = build_pair_records(deepdrid_dir, splits["val"])

    train_ds = QHatTrainingDataset(train_recs, input_size=input_size, is_train=True)
    val_ds   = QHatTrainingDataset(val_recs, input_size=input_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = QEstimator(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                val_loss += loss_fn(preds, targets).item() * len(targets)
        val_loss /= max(1, len(val_ds))

        print(f"  QEstimator Ep {ep+1:2d}/{epochs} | Train MSE={train_loss:.5f} | Val MSE={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"  ✓ QEstimator training complete! Checkpoint saved at: {ckpt_path}")

    # Run degradation calibration and save results
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    sample_imgs = [Image.open(r["path1"]).convert("RGB") for r in val_recs[:15]]
    real_labels = [r["q1"] for r in val_recs[:15]]
    calib_json = os.path.join(output_dir, "degradation_params.json")
    calibrate_and_verify(model, sample_imgs, real_labels, output_path=calib_json, device=device)
    print(f"  ✓ Degradation calibration complete → {calib_json}")

    return ckpt_path
