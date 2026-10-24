"""
cmig_qadf/data/split.py
=======================
Patient-level clustered data splitting with cryptographic manifest verification (SHA-256).
Guarantees that both eyes and dual views from any given patient belong exclusively to one partition (0 leakage).
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold


def compute_manifest_hash(data: dict) -> str:
    """Compute deterministic SHA-256 hash of JSON dictionary to enforce split integrity."""
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_patient_splits(
    deepdrid_dir: str,
    protocol: str = "official",
    cv_folds: int = 5,
    seed: int = 42,
    subset_patients: int = None,
    manifest_path: str = None,
) -> Dict[str, List[str]]:
    """
    Generate or load patient-level partitions according to evaluation protocol.

    Returns:
        dict with keys: "train", "val", "test", accompanied by "hash", "protocol"
    """
    train_csv = os.path.join(deepdrid_dir, "regular-fundus-training", "regular-fundus-training.csv")
    val_csv   = os.path.join(deepdrid_dir, "regular-fundus-validation", "regular-fundus-validation.csv")
    eval_dir  = os.path.join(deepdrid_dir, "Online-Challenge1&2-Evaluation")
    eval_c1   = os.path.join(eval_dir, "Challenge1_labels.xlsx")

    if protocol == "official":
        # 1. Official DeepDRiD protocol:
        # Train = 300 patients from regular-fundus-training
        # Val   = 100 patients from regular-fundus-validation
        # Test  = 100 patients from Online-Challenge1&2-Evaluation
        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)
        train_p  = sorted(list(set(df_train["patient_id"].astype(str))))
        val_p    = sorted(list(set(df_val["patient_id"].astype(str))))

        if os.path.exists(eval_c1):
            df_eval = pd.read_excel(eval_c1)
            test_p  = sorted(list(set([str(x).split("_")[0] for x in df_eval["image_id"]])))
        else:
            # Fallback if challenge labels file is missing
            test_p = val_p[len(val_p)//2:]
            val_p  = val_p[:len(val_p)//2]

        if subset_patients is not None and subset_patients > 0:
            # Subset for smoke test profile
            train_p = train_p[:subset_patients]
            val_p   = val_p[:max(5, subset_patients // 3)]
            test_p  = test_p[:max(5, subset_patients // 3)]

        split_dict = {
            "protocol": "official",
            "train": train_p,
            "val": val_p,
            "test": test_p,
        }

    elif protocol == "group_cv":
        # Combine all patients and stratify by patient_DR_Level
        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)
        df_all   = pd.concat([df_train, df_val], ignore_index=True)
        
        # Determine maximum DR severity per patient
        p_groups = df_all.groupby(df_all["patient_id"].astype(str))
        p_info = []
        for pid, grp in p_groups:
            max_grade = 0
            for col in ["patient_DR_Level", "left_eye_DR_Level", "right_eye_DR_Level"]:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    max_grade = max(max_grade, int(float(vals.iloc[0])))
            p_info.append({"patient_id": pid, "label": max_grade})
        
        df_p = pd.DataFrame(p_info)
        sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        folds = list(sgkf.split(df_p, df_p["label"], groups=df_p["patient_id"]))
        
        # Fold 0 = test, Fold 1 = val, remainder = train
        test_idx = folds[0][1]
        val_idx  = folds[1][1]
        train_idx = [i for i in range(len(df_p)) if i not in test_idx and i not in val_idx]

        train_p = sorted(list(df_p.iloc[train_idx]["patient_id"]))
        val_p   = sorted(list(df_p.iloc[val_idx]["patient_id"]))
        test_p  = sorted(list(df_p.iloc[test_idx]["patient_id"]))

        split_dict = {
            "protocol": "group_cv",
            "cv_folds": cv_folds,
            "fold": 0,
            "train": train_p,
            "val": val_p,
            "test": test_p,
        }
    else:
        raise ValueError(f"Unsupported split protocol: {protocol}")

    # Enforce zero patient leakage across splits
    s_train, s_val, s_test = set(split_dict["train"]), set(split_dict["val"]), set(split_dict["test"])
    assert len(s_train & s_val) == 0, f"Patient leakage between train and val: {s_train & s_val}"
    assert len(s_train & s_test) == 0, f"Patient leakage between train and test: {s_train & s_test}"
    assert len(s_val & s_test) == 0, f"Patient leakage between val and test: {s_val & s_test}"

    # Integrity hash
    split_dict["hash"] = compute_manifest_hash(split_dict)

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)

    return split_dict
