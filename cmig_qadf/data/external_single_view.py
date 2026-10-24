"""
cmig_qadf/data/external_single_view.py
======================================
External dataset loaders (APTOS 2019, IDRiD, DDR) for single-view stress testing (Appendix Table S6).
Protocol convention: Single image fed to View 1; View 2 is an identical duplicate with q2 = q1.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from cmig_qadf.data.adapter import get_transforms


class SingleViewStressDataset(Dataset):
    """
    Dataset wrapping single fundus images into pseudo-dual-view pairs (View 1 = original, View 2 = duplicate).
    """
    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        img = Image.open(item["path"]).convert("RGB")
        
        # View 1 and View 2 share identical input image
        img1 = self.transform(img) if self.transform else img
        img2 = self.transform(img) if self.transform else img
        
        return {
            "img1": img1,
            "img2": img2,
            "q1": torch.tensor([0.8], dtype=torch.float32),
            "q2": torch.tensor([0.8], dtype=torch.float32),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
            "patient_id": item["id"],
            "eye_id": "single",
            "pair_id": item["id"],
            "path1": item["path"],
            "path2": item["path"],
        }


def build_aptos_samples(aptos_dir: str) -> List[Dict]:
    csv_path = os.path.join(aptos_dir, "train.csv")
    img_dir = os.path.join(aptos_dir, "train_images")
    if not os.path.exists(csv_path): return []
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        img_id = str(row["id_code"])
        p = os.path.join(img_dir, f"{img_id}.png")
        if not os.path.exists(p):
            p = os.path.join(img_dir, f"{img_id}.jpg")
        if os.path.exists(p):
            samples.append({"id": img_id, "path": p, "label": int(row["diagnosis"])})
    return samples


def build_idrid_samples(idrid_dir: str) -> List[Dict]:
    samples = []
    gt_dir = os.path.join(idrid_dir, "2. Groundtruths")
    if not os.path.exists(gt_dir): return []
    
    for split_csv, img_sub in [
        ("a. IDRiD_Disease Grading_Training Labels.csv", "a. Training Set"),
        ("b. IDRiD_Disease Grading_Testing Labels.csv", "b. Testing Set")
    ]:
        c_path = os.path.join(gt_dir, split_csv)
        i_dir = os.path.join(idrid_dir, "1. Original Images", img_sub)
        if os.path.exists(c_path) and os.path.exists(i_dir):
            df = pd.read_csv(c_path)
            id_col = "Image name" if "Image name" in df.columns else df.columns[0]
            dr_col = "Retinopathy grade" if "Retinopathy grade" in df.columns else df.columns[1]
            for _, r in df.iterrows():
                im_name = str(r[id_col]).strip()
                p = os.path.join(i_dir, f"{im_name}.jpg")
                if os.path.exists(p):
                    samples.append({"id": im_name, "path": p, "label": int(r[dr_col])})
    return samples


def build_ddr_samples(ddr_dir: str) -> List[Dict]:
    samples = []
    for sp in ["train", "valid", "test"]:
        txt_path = os.path.join(ddr_dir, f"{sp}.txt")
        img_dir = os.path.join(ddr_dir, sp)
        if os.path.exists(txt_path) and os.path.exists(img_dir):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        fname, lbl_str = parts[0], parts[1]
                        try:
                            lbl = int(lbl_str)
                            if lbl <= 4: # Retain valid grades 0 to 4
                                p = os.path.join(img_dir, fname)
                                if os.path.exists(p):
                                    samples.append({"id": fname, "path": p, "label": lbl})
                        except ValueError:
                            pass
    return samples


def get_external_loaders(
    root_dir: str = ".",
    image_size: int = 448,
    batch_size: int = 16,
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """Create data loaders for single-view stress evaluation across external cohorts."""
    tf = get_transforms(image_size, is_train=False, use_clahe=True)
    loaders = {}

    aptos_samples = build_aptos_samples(os.path.join(root_dir, "Aptos"))
    if aptos_samples:
        loaders["APTOS"] = DataLoader(
            SingleViewStressDataset(aptos_samples, transform=tf),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    idrid_samples = build_idrid_samples(os.path.join(root_dir, "IDRiD", "B. Disease Grading"))
    if idrid_samples:
        loaders["IDRiD"] = DataLoader(
            SingleViewStressDataset(idrid_samples, transform=tf),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    ddr_samples = build_ddr_samples(os.path.join(root_dir, "DDR", "DR_grading"))
    if ddr_samples:
        loaders["DDR"] = DataLoader(
            SingleViewStressDataset(ddr_samples, transform=tf),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return loaders
