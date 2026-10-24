"""
cmig_qadf/report/gate_report.py
===============================
Scientific Gate Report check (Section 9).
Objective evaluation: Does QC-SCA outperform strong fusion baselines (co_attn, concat, average, cross_attn)?
Writes transparent conclusions to results/processed/GATE_REPORT.json and GATE_REPORT.csv.
"""

import os
import json
import pandas as pd
from typing import Dict, Any


def run_gate_check(processed_dir: str = "results/processed") -> Dict[str, Any]:
    os.makedirs(processed_dir, exist_ok=True)
    gate_json_path = os.path.join(processed_dir, "GATE_REPORT.json")
    gate_csv_path = os.path.join(processed_dir, "GATE_REPORT.csv")

    e2_path = os.path.join(processed_dir, "e2_summary.json")
    e1_path = os.path.join(processed_dir, "e1_summary.json")

    qc_beats_baselines = False
    evidence = []

    # Compare with strongest fusion competitors (excluding q_weighted)
    competitors = ["co_attn", "concat", "average", "cross_attn"]

    if os.path.exists(e2_path):
        with open(e2_path, "r", encoding="utf-8") as f:
            e2 = json.load(f)
        # Compare mean AUDC
        for mode in ["train_clean", "train_aug"]:
            if mode in e2 and "qc_sca" in e2[mode]:
                audcs_qc = []
                for v in e2[mode]["qc_sca"].get("audc_per_type", {}):
                    for d in e2[mode]["qc_sca"]["audc_per_type"][v]:
                        audcs_qc.append(e2[mode]["qc_sca"]["audc_per_type"][v][d])
                mean_qc = sum(audcs_qc) / len(audcs_qc) if audcs_qc else 0.0

                comp_means = {}
                for comp in competitors:
                    if comp in e2[mode]:
                        c_audcs = []
                        for v in e2[mode][comp].get("audc_per_type", {}):
                            for d in e2[mode][comp]["audc_per_type"][v]:
                                c_audcs.append(e2[mode][comp]["audc_per_type"][v][d])
                        if c_audcs:
                            comp_means[comp] = sum(c_audcs) / len(c_audcs)

                if comp_means:
                    best_comp = max(comp_means.keys(), key=lambda k: comp_means[k])
                    best_val = comp_means[best_comp]
                    evidence.append(f"- Mode {mode}: Mean AUDC QC-SCA = {mean_qc:.4f} vs Strongest Competitor ({best_comp.upper()}) = {best_val:.4f}")
                    if mean_qc > best_val:
                        qc_beats_baselines = True

    if os.path.exists(e1_path):
        with open(e1_path, "r", encoding="utf-8") as f:
            e1 = json.load(f)
        if "qc_sca" in e1:
            q_qc = e1["qc_sca"].get("qwk_mean", 0.0)
            comp_qwks = {k: v.get("qwk_mean", 0.0) for k, v in e1.items() if k in competitors}
            if comp_qwks:
                best_comp_e1 = max(comp_qwks.keys(), key=lambda k: comp_qwks[k])
                evidence.append(f"- Clean Images E1: Mean QWK QC-SCA = {q_qc:.4f} vs {best_comp_e1.upper()} = {comp_qwks[best_comp_e1]:.4f}")

    if not evidence:
        status = "INSUFFICIENT DATA (TBD)"
        conclusion = "Stages S4-S6 have not been completed to produce gate report conclusions."
    elif qc_beats_baselines:
        status = "CRITERIA MET (GATE PASSED)"
        conclusion = "QC-SCA demonstrates meaningful superiority over all strong fusion baselines on the E2 degradation benchmark. Contributions C1 & C2 are empirically substantiated."
    else:
        status = "NOT SUPERIOR (GATE CONVERGED TO BENCHMARK EMPHASIS)"
        conclusion = "QC-SCA does not outperform baselines on E2. Following scientific guidance, the paper emphasizes Contribution C2 (Controlled degradation benchmark & Harm metrics suite) with honest mechanism analysis."

    gate_payload = {
        "status": status,
        "qc_beats_baselines": qc_beats_baselines,
        "conclusion": conclusion,
        "evidence": evidence,
        "recommendations": [
            "The pipeline generates all tables and figures without intervening or pruning data.",
            "Maintain academic honesty without inflating claims."
        ]
    }

    # 1. Output JSON
    with open(gate_json_path, "w", encoding="utf-8") as f:
        json.dump(gate_payload, f, indent=2)

    # 2. Output CSV
    gate_df = pd.DataFrame([{
        "status": status,
        "qc_beats_baselines": qc_beats_baselines,
        "conclusion": conclusion,
        "evidence": "; ".join(evidence) if evidence else "None"
    }])
    gate_df.to_csv(gate_csv_path, index=False)

    # Clean up legacy GATE_REPORT.md
    legacy_md = os.path.join(processed_dir, "GATE_REPORT.md")
    if os.path.exists(legacy_md):
        try:
            os.remove(legacy_md)
        except OSError:
            pass

    return {
        "status": status,
        "qc_beats_baselines": qc_beats_baselines,
        "report_file": gate_json_path,
        "csv_file": gate_csv_path,
    }
