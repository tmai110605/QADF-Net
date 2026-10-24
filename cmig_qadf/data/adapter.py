"""
cmig_qadf/data/adapter.py
=========================
DeepDRiD data adapter wrapping dataset into a unified schema across all 3 sets:
Train (300 patients), Val (100 patients), and Test Challenge (100 patients).
Each sample returns: (img1, img2, q1, q2, label, patient_id, eye_id, pair_id, path1, path2).
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from typing import Optional, Callable, Dict, List, Tuple


def extract_quality_score(row: pd.Series) -> float:
    """Extract normalized optical quality score ∈ [0.05, 1.0]."""
    if "Clarity" in row and "Field definition" in row and "Artifact" in row:
        try:
            c = float(row["Clarity"])
            f = float(row["Field definition"])
            a = float(row["Artifact"])
            return float(max(0.05, min(1.0, (c + f + (10.0 - a)) / 30.0)))
        except (ValueError, TypeError):
            pass
    raw = float(row.get("Overall quality", 0.0))
    return 0.8 if raw == 0.0 else 0.35


class CLAHETransform:
    """Contrast Limited Adaptive Histogram Equalization on Luminance (L-channel)."""
    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img_bgr_out = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
        img_rgb_out = cv2.cvtColor(img_bgr_out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb_out)


def get_transforms(image_size: int = 448, is_train: bool = True, use_clahe: bool = True, clahe_clip: float = 2.0):
    ops = []
    if is_train:
        ops.extend([
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    else:
        ops.extend([
            transforms.Resize(size=(image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        ])

    if use_clahe:
        ops.append(CLAHETransform(clip_limit=clahe_clip))

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


class DeepDRiDAdapterDataset(Dataset):
    """
    Dataset wrapping all DeepDRiD image pairs into a unified schema.
    """
    def __init__(
        self,
        records: List[Dict],
        transform: Optional[Callable] = None,
        degradation_fn: Optional[Callable] = None,
        cached_images: Optional[Dict[str, Tuple[Image.Image, Image.Image]]] = None,
    ):
        self.records = records
        self.transform = transform
        self.degradation_fn = degradation_fn
        self.cached_images = cached_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        if self.cached_images is not None and rec["pair_id"] in self.cached_images:
            im1, im2 = self.cached_images[rec["pair_id"]]
            img1 = im1.copy()
            img2 = im2.copy()
        else:
            with Image.open(rec["path1"]) as im1:
                img1 = im1.convert("RGB")
            with Image.open(rec["path2"]) as im2:
                img2 = im2.convert("RGB")
        q1 = float(rec["q1"])
        q2 = float(rec["q2"])

        # If evaluation degradation function is provided (E2, E3, E4)
        if self.degradation_fn is not None:
            img1, img2, q1, q2 = self.degradation_fn(
                img1=img1, img2=img2, q1=q1, q2=q2,
                sample_id=rec["pair_id"],
                patient_id=rec["patient_id"]
            )

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            "img1": img1,
            "img2": img2,
            "q1": torch.tensor([q1], dtype=torch.float32),
            "q2": torch.tensor([q2], dtype=torch.float32),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "patient_id": rec["patient_id"],
            "eye_id": rec["eye_id"],
            "pair_id": rec["pair_id"],
            "path1": rec["path1"],
            "path2": rec["path2"],
        }

    def get_labels(self) -> List[int]:
        return [r["label"] for r in self.records]


def build_pair_records(
    deepdrid_dir: str,
    patient_ids: List[str]
) -> List[Dict]:
    """
    Scan directory and pair dual-view fundus images by patient and eye.
    """
    patient_set = set(str(p) for p in patient_ids)
    all_records = []

    # 1. Scan training and validation folders
    for sub in ["regular-fundus-training", "regular-fundus-validation"]:
        csv_file = os.path.join(deepdrid_dir, sub, f"{sub}.csv")
        img_root = os.path.join(deepdrid_dir, sub, "Images")
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file)
        
        # Group by patient
        for pid, grp in df.groupby(df["patient_id"].astype(str)):
            if pid not in patient_set:
                continue
            
            # Split left (l) and right (r) eyes
            for eye in ["l", "r"]:
                m_view1 = grp["image_id"].astype(str).str.endswith(f"{eye}1")
                m_view2 = grp["image_id"].astype(str).str.endswith(f"{eye}2")

                if m_view1.any() and m_view2.any():
                    row1 = grp[m_view1].iloc[0]
                    row2 = grp[m_view2].iloc[0]

                    p1 = os.path.join(img_root, pid, f"{row1['image_id']}.jpg")
                    p2 = os.path.join(img_root, pid, f"{row2['image_id']}.jpg")

                    if os.path.exists(p1) and os.path.exists(p2):
                        # DR label
                        lbl = None
                        for c in [f"{'left' if eye=='l' else 'right'}_eye_DR_Level", "patient_DR_Level"]:
                            if c in row1 and pd.notna(row1[c]):
                                lbl = int(float(row1[c]))
                                break
                        if lbl is None:
                            continue

                        all_records.append({
                            "patient_id": pid,
                            "eye_id": eye,
                            "pair_id": f"{pid}_{eye}",
                            "path1": p1,
                            "path2": p2,
                            "q1": extract_quality_score(row1),
                            "q2": extract_quality_score(row2),
                            "label": lbl,
                        })

    # 2. Scan Online-Challenge1&2-Evaluation
    eval_dir = os.path.join(deepdrid_dir, "Online-Challenge1&2-Evaluation")
    eval_c1 = os.path.join(eval_dir, "Challenge1_labels.xlsx")
    eval_c2 = os.path.join(eval_dir, "Challenge2_labels.xlsx")
    eval_imgs = os.path.join(eval_dir, "Images")

    if os.path.exists(eval_c1) and os.path.exists(eval_c2):
        df_c1 = pd.read_excel(eval_c1)
        df_c2 = pd.read_excel(eval_c2)
        df_eval = pd.merge(df_c1, df_c2, on="image_id", how="inner")

        for img_id, row in df_eval.iterrows():
            img_name = str(row["image_id"]).strip()
            pid = img_name.split("_")[0]
            if pid not in patient_set:
                continue

            # Pair only from view 1 to avoid duplicate entries
            if img_name.endswith("1"):
                stem = img_name[:-1]
                v2_name = stem + "2"
                eye = "l" if "_l" in img_name else "r"

                mask2 = df_eval["image_id"].astype(str).str.strip() == v2_name
                if mask2.any():
                    row2 = df_eval[mask2].iloc[0]
                    p1 = os.path.join(eval_imgs, pid, img_name + ".jpg")
                    p2 = os.path.join(eval_imgs, pid, v2_name + ".jpg")

                    if os.path.exists(p1) and os.path.exists(p2):
                        all_records.append({
                            "patient_id": pid,
                            "eye_id": eye,
                            "pair_id": f"{pid}_{eye}",
                            "path1": p1,
                            "path2": p2,
                            "q1": extract_quality_score(row),
                            "q2": extract_quality_score(row2),
                            "label": int(row["DR_Levels"]),
                        })

    return all_records


def build_dataloaders(
    deepdrid_dir: str,
    splits: Dict[str, List[str]],
    image_size: int = 448,
    batch_size: int = 8,
    num_workers: int = 4,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    degradation_fn: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """Create data loaders for train, val, and test partitions."""
    train_recs = build_pair_records(deepdrid_dir, splits["train"])
    val_recs   = build_pair_records(deepdrid_dir, splits["val"])
    test_recs  = build_pair_records(deepdrid_dir, splits["test"])

    train_tf = get_transforms(image_size, is_train=True, use_clahe=use_clahe, clahe_clip=clahe_clip)
    eval_tf  = get_transforms(image_size, is_train=False, use_clahe=use_clahe, clahe_clip=clahe_clip)

    train_ds = DeepDRiDAdapterDataset(train_recs, transform=train_tf)
    val_ds   = DeepDRiDAdapterDataset(val_recs, transform=eval_tf)
    test_ds  = DeepDRiDAdapterDataset(test_recs, transform=eval_tf, degradation_fn=degradation_fn)

    # Balanced class sampler for training
    labels = train_ds.get_labels()
    counts = np.bincount(labels, minlength=5).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
    }
