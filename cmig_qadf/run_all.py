"""
cmig_qadf/run_all.py
====================
Master DAG Orchestrator for the QADF-Net study (CMIG).
Executes sequentially or via DAG from S0 through S8:
  - S0: Environment verification, patient split generation, unit test suite
  - S1: Train and freeze q_hat quality estimator, degradation calibration
  - S2: Train all models (method x backbone x seed x train mode)
  - S3: Predict on clean test set (save sample predictions.parquet)
  - S4: Evaluate under controlled degradation benchmark (E2, E3, E4)
  - S5: Specialized analyses (E5, E6, Proposition 1, E7, E9, E10, single-view)
  - S6: Statistical inference: Clustered bootstrap by patient, Holm-Bonferroni correction
  - S7: Scientific Gate Report evaluation
  - S8: Export all 11 tables (.csv, .json), figures (.png, .pdf), and summary report
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time
import json
import yaml
import argparse
import subprocess
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Any

# Ensure cmig_qadf package is discoverable from root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cmig_qadf.data.split import get_patient_splits
from cmig_qadf.train.train_qhat import train_q_estimator
from cmig_qadf.train.trainer import train_one
from cmig_qadf.data.adapter import build_dataloaders, build_pair_records
from cmig_qadf.models.dualview_model import DualViewGradingModel
from cmig_qadf.models.registry import build_model
from cmig_qadf.models.q_estimator import QEstimator
from cmig_qadf.eval.predict import run_prediction
from cmig_qadf.eval.degradation_eval import evaluate_degradation_grid
from cmig_qadf.eval.proposition1 import verify_proposition1_empirically
from cmig_qadf.eval.attention_analysis import analyze_attention_e7
from cmig_qadf.eval.efficiency import measure_latency_and_vram
from cmig_qadf.eval.error_analysis import analyze_errors_e10
from cmig_qadf.experiments.e1_main import get_jobs as get_e1_jobs, aggregate_e1
from cmig_qadf.experiments.e2_degradation import get_jobs as get_e2_jobs, aggregate_e2
from cmig_qadf.experiments.e3_strata import aggregate_e3
from cmig_qadf.experiments.e4_qsource import get_jobs as get_e4_jobs, aggregate_e4
from cmig_qadf.experiments.e5_mechanisms import get_jobs as get_e5_jobs, aggregate_e5
from cmig_qadf.experiments.e6_tokens import get_jobs as get_e6_jobs, aggregate_e6
from cmig_qadf.experiments.e8_backbones import get_jobs as get_e8_jobs, aggregate_e8
from cmig_qadf.report.gate_report import run_gate_check
from cmig_qadf.report.make_tables import generate_all_tables
from cmig_qadf.report.make_figures import generate_all_figures
from cmig_qadf.report.summary import generate_results_summary


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(base: dict, update: dict) -> dict:
    res = base.copy()
    for k, v in update.items():
        if isinstance(v, dict) and k in res and isinstance(res[k], dict):
            res[k] = merge_dicts(res[k], v)
        else:
            res[k] = v
    return res


class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.profile = args.profile
        self.root_dir = ROOT_DIR

        # Load configuration
        default_cfg_path = os.path.join(self.root_dir, "cmig_qadf", "configs", "default.yaml")
        profile_cfg_path = os.path.join(self.root_dir, "cmig_qadf", "configs", "profiles", f"{self.profile}.yaml")
        
        cfg = load_yaml(default_cfg_path)
        if os.path.exists(profile_cfg_path):
            p_cfg = load_yaml(profile_cfg_path)
            cfg = merge_dicts(cfg, p_cfg)
        self.cfg = cfg
        if "data" in self.cfg and "deepdrid_dir" in self.cfg["data"]:
            self.cfg["data"]["deepdrid_dir"] = os.path.abspath(os.path.join(self.root_dir, self.cfg["data"]["deepdrid_dir"]))

        # Results directories
        self.results_dir = os.path.join(self.root_dir, "results")
        self.raw_dir = os.path.join(self.results_dir, "raw")
        self.processed_dir = os.path.join(self.results_dir, "processed")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.manifests_dir = os.path.join(self.results_dir, "manifests")
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")

        for d in [self.raw_dir, self.processed_dir, self.logs_dir, self.manifests_dir, self.checkpoints_dir]:
            os.makedirs(d, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.watermark = self.cfg.get("watermark", "") if self.profile == "smoke" else ""
        self.failed_jobs = []

        # Load splits from existing manifest if present
        manifest_filename = f"split_manifest_{self.profile}.json" if self.profile == "smoke" else "split_manifest.json"
        manifest_path = os.path.join(self.manifests_dir, manifest_filename)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.splits = json.load(f)
        else:
            deepdrid_dir = os.path.join(self.root_dir, self.cfg["data"]["deepdrid_dir"])
            protocol = self.cfg["data"].get("protocol", "official")
            subset_patients = self.cfg["data"].get("subset_patients", None) if self.profile == "smoke" else None
            self.splits = get_patient_splits(
                deepdrid_dir=deepdrid_dir,
                protocol=protocol,
                subset_patients=subset_patients,
                manifest_path=manifest_path,
            )

    def log_failed(self, job_id: str, error: str):
        self.failed_jobs.append({"job_id": job_id, "error": str(error)})
        fail_file = os.path.join(self.logs_dir, "failed_jobs.json")
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(self.failed_jobs, f, indent=2)

    # -------------------------------------------------------------------------
    # STAGE S0: Environment, Split, and Unit Tests
    # -------------------------------------------------------------------------
    def run_stage_s0(self):
        print(f"\n{'='*70}\n[STAGE S0] ENVIRONMENT VERIFICATION, PATIENT SPLITS & UNIT TESTS\n{'='*70}")
        # 1. Run unit tests
        test_dir = os.path.join(self.root_dir, "cmig_qadf", "tests")
        print("  -> Executing mandatory unit test suite...")
        res = subprocess.run([sys.executable, "-m", "pytest", test_dir, "-q"], cwd=self.root_dir)
        if res.returncode != 0:
            print("  [ERROR] Unit test suite failed! Halting pipeline per Section 12 requirements.")
            sys.exit(1)
        print("  [OK] All unit tests passed successfully.")

        # 2. Generate patient-level splits with SHA-256 hash manifest
        manifest_filename = f"split_manifest_{self.profile}.json" if self.profile == "smoke" else "split_manifest.json"
        manifest_path = os.path.join(self.manifests_dir, manifest_filename)
        deepdrid_dir = os.path.join(self.root_dir, self.cfg["data"]["deepdrid_dir"])
        protocol = self.cfg["data"].get("protocol", "official")
        subset_patients = self.cfg["data"].get("subset_patients", None) if self.profile == "smoke" else None

        splits = get_patient_splits(
            deepdrid_dir=deepdrid_dir,
            protocol=protocol,
            subset_patients=subset_patients,
            manifest_path=manifest_path,
        )
        print(f"  [OK] Split completed (Protocol: {protocol}): {len(splits['train'])} Train, {len(splits['val'])} Val, {len(splits['test'])} Test.")
        print(f"  [OK] Split Hash: {splits['hash']}")
        self.splits = splits
        return splits

    # -------------------------------------------------------------------------
    # STAGE S1: Train QEstimator
    # -------------------------------------------------------------------------
    def run_stage_s1(self):
        print(f"\n{'='*70}\n[STAGE S1] TRAIN AND FREEZE Q-ESTIMATOR & DEGRADATION CALIBRATION\n{'='*70}")
        deepdrid_dir = os.path.join(self.root_dir, self.cfg["data"]["deepdrid_dir"])
        epochs = self.cfg.get("quality", {}).get("estimator_epochs", 2 if self.profile == "smoke" else 15)
        in_size = self.cfg.get("quality", {}).get("estimator_input_size", 224)

        q_ckpt = train_q_estimator(
            deepdrid_dir=deepdrid_dir,
            splits=self.splits,
            epochs=epochs,
            input_size=in_size,
            device=self.device,
            checkpoint_dir=self.checkpoints_dir,
            output_dir=self.processed_dir,
        )
        self.q_estimator_ckpt = q_ckpt
        return q_ckpt

    # -------------------------------------------------------------------------
    # STAGE S2: Training Models
    # -------------------------------------------------------------------------
    def run_stage_s2(self):
        print(f"\n{'='*70}\n[STAGE S2] TRAIN ALL MODELS (DAG WORKERS)\n{'='*70}")
        # Collect all training jobs from E1, E2, E4, E5, E6, E8
        all_jobs = []
        all_jobs.extend(get_e1_jobs(self.cfg, profile=self.profile))
        if self.profile != "smoke":
            all_jobs.extend(get_e2_jobs(self.cfg, profile=self.profile))
            all_jobs.extend(get_e4_jobs(self.cfg, profile=self.profile))
            all_jobs.extend(get_e5_jobs(self.cfg, profile=self.profile))
            all_jobs.extend(get_e6_jobs(self.cfg, profile=self.profile))
            all_jobs.extend(get_e8_jobs(self.cfg, profile=self.profile))

        print(f"  Total training jobs to process in S2: {len(all_jobs)}")

        for idx, job in enumerate(all_jobs):
            job["splits"] = self.splits
            job["q_estimator_ckpt"] = getattr(self, "q_estimator_ckpt", None)
            job["raw_dir"] = os.path.join(self.raw_dir, "S2")

            print(f"\n[{idx+1}/{len(all_jobs)}] Job: {job['job_id']}")
            try:
                train_one(job, force=self.args.force)
            except Exception as e:
                print(f"  [ERROR] Job {job['job_id']} encountered error: {e}")
                self.log_failed(job["job_id"], str(e))
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE S3: Predict on Clean
    # -------------------------------------------------------------------------
    def run_stage_s3(self):
        print(f"\n{'='*70}\n[STAGE S3] PREDICT ON CLEAN VAL/TEST (SAVE SAMPLE PARQUET)\n{'='*70}")
        s2_dir = os.path.join(self.raw_dir, "S2")
        s3_dir = os.path.join(self.raw_dir, "S3")
        os.makedirs(s3_dir, exist_ok=True)

        deepdrid_dir = os.path.join(self.root_dir, self.cfg["data"]["deepdrid_dir"])
        loaders = build_dataloaders(
            deepdrid_dir=deepdrid_dir,
            splits=self.splits,
            image_size=self.cfg["data"].get("image_size", 448),
            batch_size=16,
            num_workers=0
        )
        test_loader = loaders["test"]

        # Iterate over all completed checkpoints
        if not os.path.exists(s2_dir):
            return

        q_estimator = None
        q_ckpt = getattr(self, "q_estimator_ckpt", os.path.join(self.checkpoints_dir, "q_estimator_best.pt"))
        if q_ckpt and os.path.exists(q_ckpt):
            q_estimator = QEstimator(pretrained=False)
            q_estimator.load_state_dict(torch.load(q_ckpt, map_location=self.device))
            q_estimator.to(self.device).eval()

        for job_folder in os.listdir(s2_dir):
            job_path = os.path.join(s2_dir, job_folder)
            ckpt_path = os.path.join(job_path, "best_checkpoint.pt")
            cfg_path = os.path.join(job_path, "config.yaml")
            parquet_path = os.path.join(s3_dir, f"{job_folder}.parquet")

            if not os.path.exists(ckpt_path) or not os.path.exists(cfg_path):
                continue

            if not self.args.force and os.path.exists(parquet_path):
                continue

            with open(cfg_path, "r", encoding="utf-8") as f:
                job_cfg = yaml.safe_load(f)

            try:
                # Initialize model and load weights
                model = build_model(job_cfg, q_estimator=q_estimator)
                model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
                model.to(self.device)

                print(f"  -> Predicting samples for: {job_folder}")
                run_prediction(model, test_loader, device=self.device, output_path=parquet_path)
            except Exception as e:
                print(f"  [ERROR] Prediction error for job {job_folder}: {e}")
                self.log_failed(f"s3_{job_folder}", str(e))
            finally:
                if "model" in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE S4: Controlled Degradation Evaluation
    # -------------------------------------------------------------------------
    def run_stage_s4(self):
        print(f"\n{'='*70}\n[STAGE S4] CONTROLLED DEGRADATION BENCHMARK EVALUATION (E2, E3, E4)\n{'='*70}")
        s2_dir = os.path.join(self.raw_dir, "S2")
        s4_dir = os.path.join(self.raw_dir, "S4")
        os.makedirs(s4_dir, exist_ok=True)

        deepdrid_dir = os.path.join(self.root_dir, self.cfg["data"]["deepdrid_dir"])
        test_records = build_pair_records(deepdrid_dir, self.splits["test"])

        deg_types = self.cfg.get("eval", {}).get("degradation_types", None)
        levels = self.cfg.get("eval", {}).get("degradation_levels", [0, 1, 2, 3, 4, 5])

        if not os.path.exists(s2_dir):
            return

        q_estimator = None
        q_ckpt = getattr(self, "q_estimator_ckpt", os.path.join(self.checkpoints_dir, "q_estimator_best.pt"))
        if q_ckpt and os.path.exists(q_ckpt):
            q_estimator = QEstimator(pretrained=False)
            q_estimator.load_state_dict(torch.load(q_ckpt, map_location=self.device))
            q_estimator.to(self.device).eval()

        for job_folder in os.listdir(s2_dir):
            ckpt_path = os.path.join(s2_dir, job_folder, "best_checkpoint.pt")
            cfg_path = os.path.join(s2_dir, job_folder, "config.yaml")
            out_json = os.path.join(s4_dir, f"{job_folder}_degradations.json")

            if not os.path.exists(ckpt_path) or not os.path.exists(cfg_path):
                continue

            if not self.args.force and os.path.exists(out_json):
                continue

            with open(cfg_path, "r", encoding="utf-8") as f:
                job_cfg = yaml.safe_load(f)

            try:
                model = build_model(job_cfg, q_estimator=q_estimator)
                model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
                model.to(self.device)

                print(f"  -> Running E2 degradation grid for: {job_folder}")
                evaluate_degradation_grid(
                    model=model,
                    test_records=test_records,
                    deg_types=deg_types,
                    levels=levels,
                    image_size=self.cfg["data"].get("image_size", 448),
                    device=self.device,
                    output_path=out_json
                )
            except Exception as e:
                print(f"  [ERROR] Degradation evaluation error for job {job_folder}: {e}")
                self.log_failed(f"s4_{job_folder}", str(e))
            finally:
                if "model" in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE S5: Specialized Analyses (Proposition 1, E7, E9, E10)
    # -------------------------------------------------------------------------
    def run_stage_s5(self):
        print(f"\n{'='*70}\n[STAGE S5] SPECIALIZED ANALYSES: PROPOSITION 1, E7, E9, E10\n{'='*70}")
        deepdrid_dir = os.path.join(self.root_dir, self.cfg["data"]["deepdrid_dir"])
        loaders = build_dataloaders(
            deepdrid_dir=deepdrid_dir,
            splits=self.splits,
            image_size=self.cfg["data"].get("image_size", 448),
            batch_size=16,
            num_workers=0
        )
        test_loader = loaders["test"]

        # 1. Empirical verification of Proposition 1
        qc_ckpt = os.path.join(self.raw_dir, "S2", "e1_qc_sca_efficientnet_b4_seed42_clean", "best_checkpoint.pt")
        qc_cfg_path = os.path.join(self.raw_dir, "S2", "e1_qc_sca_efficientnet_b4_seed42_clean", "config.yaml")
        if os.path.exists(qc_cfg_path) and os.path.exists(qc_ckpt):
            with open(qc_cfg_path, "r", encoding="utf-8") as f:
                qc_cfg = yaml.safe_load(f)
            model_qc = build_model(qc_cfg)
            model_qc.load_state_dict(torch.load(qc_ckpt, map_location=self.device), strict=False)
        else:
            model_qc = DualViewGradingModel(backbone_name="efficientnet_b4", fusion_type="qc_sca", pretrained=False, grid_size=self.cfg["model"].get("grid_size", 14))
        model_qc.to(self.device)
        prop_out = os.path.join(self.processed_dir, "proposition1_summary.json")
        verify_proposition1_empirically(model_qc, test_loader, device=self.device, output_path=prop_out)
        print("  [OK] Proposition 1 empirical verification complete.")

        # 2. Computational efficiency E9
        e9_out = os.path.join(self.processed_dir, "e9_efficiency_summary.json")
        methods_e9 = ["qc_sca", "concat", "average", "cross_attn"]
        res_e9 = {}
        e9_in_size = self.cfg["data"].get("image_size", 448)
        n_warm = 1 if self.profile == "smoke" else 3
        n_runs = 3 if self.profile == "smoke" else 10
        for m in methods_e9:
            m_obj = DualViewGradingModel(backbone_name="efficientnet_b4", fusion_type=m, pretrained=False, grid_size=self.cfg["model"].get("grid_size", 14))
            res_e9[m] = measure_latency_and_vram(m_obj, input_size=e9_in_size, num_warmup=n_warm, num_runs=n_runs, device=self.device)
        with open(e9_out, "w", encoding="utf-8") as f:
            json.dump(res_e9, f, indent=2)
        print("  [OK] E9 efficiency measurements complete.")

        # 3. Error analysis E10
        s3_dir = os.path.join(self.raw_dir, "S3")
        e10_csv = os.path.join(self.processed_dir, "e10_cases.csv")
        files_s3 = [f for f in os.listdir(s3_dir) if f.endswith(".parquet")] if os.path.exists(s3_dir) else []
        if files_s3:
            qc_f = [f for f in files_s3 if "qc_sca" in f]
            v1_f = [f for f in files_s3 if "single_v1" in f]
            v2_f = [f for f in files_s3 if "single_v2" in f]
            if qc_f and v1_f:
                df_qc = pd.read_parquet(os.path.join(s3_dir, qc_f[0]))
                df_v1 = pd.read_parquet(os.path.join(s3_dir, v1_f[0]))
                df_v2 = pd.read_parquet(os.path.join(s3_dir, v2_f[0])) if v2_f else df_v1
                analyze_errors_e10(df_qc, df_v1, df_v2, output_csv_path=e10_csv)
                print("  [OK] E10 error analysis complete.")

    # -------------------------------------------------------------------------
    # STAGE S6: Statistics & Aggregation
    # -------------------------------------------------------------------------
    def run_stage_s6(self):
        print(f"\n{'='*70}\n[STAGE S6] STATISTICAL INFERENCE & AGGREGATION (BOOTSTRAP & HOLM)\n{'='*70}")
        s3_dir = os.path.join(self.raw_dir, "S3")
        s4_dir = os.path.join(self.raw_dir, "S4")
        reps = self.cfg.get("eval", {}).get("bootstrap_reps", 50 if self.profile == "smoke" else 2000)

        # Aggregate E1
        aggregate_e1(s3_dir, self.processed_dir, self.splits, bootstrap_reps=reps)
        # Aggregate E2
        aggregate_e2(s4_dir, self.processed_dir)
        # Aggregate E3
        aggregate_e3(s3_dir, self.processed_dir)
        # Aggregate E4, E5, E6, E8
        aggregate_e4(s3_dir, self.processed_dir)
        aggregate_e5(s3_dir, self.processed_dir)
        aggregate_e6(s3_dir, self.processed_dir)
        aggregate_e8(s3_dir, self.processed_dir)
        print("  [OK] Statistical paired testing and experiment aggregation complete.")

    # -------------------------------------------------------------------------
    # STAGE S7: Scientific Gate Report
    # -------------------------------------------------------------------------
    def run_stage_s7(self):
        print(f"\n{'='*70}\n[STAGE S7] SCIENTIFIC GATE REPORT EVALUATION\n{'='*70}")
        gate_res = run_gate_check(self.processed_dir)
        print(f"  Gate Status: {gate_res['status']}")
        print(f"  Report written to: {gate_res['report_file']}")

    # -------------------------------------------------------------------------
    # STAGE S8: Make All Tables, Figures, and Summary
    # -------------------------------------------------------------------------
    def run_stage_s8(self):
        print(f"\n{'='*70}\n[STAGE S8] EXPORT ALL TABLES, FIGURES & SUMMARY REPORT\n{'='*70}")
        generate_all_tables(results_dir=self.results_dir, profile=self.profile, watermark=self.watermark)
        generate_all_figures(results_dir=self.results_dir, profile=self.profile, watermark=self.watermark)
        generate_results_summary(results_dir=self.results_dir, docs_dir=os.path.join(self.root_dir, "cmig_qadf", "docs"), profile=self.profile)
        print(f"\n{'='*70}\n[SUCCESS] ENTIRE PIPELINE FOR PROFILE '{self.profile.upper()}' COMPLETED SUCCESSFULLY!\n{'='*70}")

    # -------------------------------------------------------------------------
    # DRY-RUN: Resource & Job Estimation
    # -------------------------------------------------------------------------
    def run_dry_run(self):
        print(f"\n{'='*70}\n[DRY RUN] JOB EXECUTION PLAN & RESOURCE ESTIMATION (PROFILE: {self.profile.upper()})\n{'='*70}")
        e1_jobs = get_e1_jobs(self.cfg, profile=self.profile)
        e2_jobs = get_e2_jobs(self.cfg, profile=self.profile) if self.profile != "smoke" else []
        e4_jobs = get_e4_jobs(self.cfg, profile=self.profile) if self.profile != "smoke" else []
        e5_jobs = get_e5_jobs(self.cfg, profile=self.profile) if self.profile != "smoke" else []
        e6_jobs = get_e6_jobs(self.cfg, profile=self.profile) if self.profile != "smoke" else []
        e8_jobs = get_e8_jobs(self.cfg, profile=self.profile) if self.profile != "smoke" else []

        counts = {
            "S0_Split_and_Tests": 1,
            "S1_QEstimator_Training": 1,
            "S2_E1_Main_Jobs": len(e1_jobs),
            "S2_E2_Aug_Jobs": len(e2_jobs),
            "S2_E4_QSource_Jobs": len(e4_jobs),
            "S2_E5_Mechanism_Jobs": len(e5_jobs),
            "S2_E6_TokenHead_Jobs": len(e6_jobs),
            "S2_E8_Backbone_Jobs": len(e8_jobs),
            "S3_Inference_Jobs": len(e1_jobs) + len(e2_jobs),
            "S4_Degradation_Grid_Jobs": len(e1_jobs) + len(e2_jobs),
            "S5_Analysis_Jobs": 4,
            "S6_Statistical_Bootstrap_Jobs": 1,
            "S7_Gate_Evaluation": 1,
            "S8_Tables_and_Figures": 17, # 11 tables + 6 figures
        }

        total_train_jobs = counts["S1_QEstimator_Training"] + sum([counts[k] for k in counts if k.startswith("S2_")])
        print("Job count breakdown by stage:")
        for stage, cnt in counts.items():
            print(f"  • {stage:32s}: {cnt:4d} jobs")

        print(f"\n-> Total model training runs: {total_train_jobs}")

        # GPU time estimation
        if self.profile == "smoke":
            est_hours = 0.15 # ~10-15 minutes total
            print(f"-> Estimated runtime for Smoke profile: ~0.15 GPU hours (< 15 min)")
        elif self.profile == "fast":
            est_hours = total_train_jobs * 0.25 # ~15 min / run
            print(f"-> Estimated runtime for Fast profile: ~{est_hours:.1f} GPU hours")
        else: # full
            est_hours = total_train_jobs * 0.35 # ~20 min / run
            print(f"-> Estimated runtime for Full profile: ~{est_hours:.1f} GPU hours (~{est_hours/24.0:.1f} GPU-days)")

        max_gpu_h = self.cfg.get("hardware", {}).get("max_gpu_hours", 48.0)
        if est_hours > max_gpu_h:
            print(f"\n[WARNING] Estimated time ({est_hours:.1f}h) exceeds maximum budget ({max_gpu_h:.1f}h)!")
            print("Recommended prioritization steps preserving scientific rigor:")
            print("  1. Reduce seeds for E6 (Token/Heads) and E8 (Backbone) from 5 to 3.")
            print("  2. Prune baseline set in train_aug mode (retaining Concat, Q-Weighted, Cross-Attn, QC-SCA).")
            print("  3. DO NOT prune E1 (clean images), E2 train_clean, or Q-Weighted Average baseline.")

        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Master Runner for QADF-Net (CMIG)")
    parser.add_argument("--profile", type=str, default="full", choices=["smoke", "fast", "full"], help="Execution profile")
    parser.add_argument("--only", type=str, default=None, help="Only execute specific stage(s) (e.g. S0,S1,S8)")
    parser.add_argument("--skip", type=str, default=None, help="Skip specific stage(s)")
    parser.add_argument("--force", action="store_true", help="Rerun skipping cache")
    parser.add_argument("--dry-run", action="store_true", help="Print job plan and estimated GPU hours, then exit")
    parser.add_argument("--gpus", type=str, default="0", help="GPU device list (e.g. 0,1)")
    parser.add_argument("--seeds", type=str, default=None, help="Override random seeds (e.g. 42,43)")

    args = parser.parse_args()

    orchestrator = Orchestrator(args)

    if args.dry_run:
        orchestrator.run_dry_run()

    stages = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    if args.only:
        active_stages = [s.strip().upper() for s in args.only.split(",")]
    else:
        active_stages = stages

    if args.skip:
        skip_list = [s.strip().upper() for s in args.skip.split(",")]
        active_stages = [s for s in active_stages if s not in skip_list]

    print(f"=== STARTING ORCHESTRATOR | Profile: {args.profile.upper()} | Stages: {active_stages} ===")

    if "S0" in active_stages: orchestrator.run_stage_s0()
    if "S1" in active_stages: orchestrator.run_stage_s1()
    if "S2" in active_stages: orchestrator.run_stage_s2()
    if "S3" in active_stages: orchestrator.run_stage_s3()
    if "S4" in active_stages: orchestrator.run_stage_s4()
    if "S5" in active_stages: orchestrator.run_stage_s5()
    if "S6" in active_stages: orchestrator.run_stage_s6()
    if "S7" in active_stages: orchestrator.run_stage_s7()
    if "S8" in active_stages: orchestrator.run_stage_s8()


if __name__ == "__main__":
    main()
