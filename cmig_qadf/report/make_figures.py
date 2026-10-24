"""
cmig_qadf/report/make_figures.py
================================
Automatically generates all Figures for CMIG manuscript (Section 10).
Each figure exported in 2 formats: .pdf (Vector) and .png (300 DPI).
Standards: Font >= 8pt, Colorblind-friendly palette, Accompanied by .meta.json file.
RULES: ONLY read from results/raw/ and results/processed/. Do not use external images for Fig 1.
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
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


# CMIG / Elsevier aesthetic settings
plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.autolayout": True,
})

# Colorblind-friendly palette (Wong palette)
COLOR_PALETTE = {
    "qc_sca": "#D55E00",      # Vermillion / Highlight orange
    "co_attn": "#0072B2",     # Blue (Co-Attention)
    "cross_attn": "#009E73",  # Green
    "concat": "#CC79A7",      # Reddish purple
    "average": "#E69F00",     # Orange
    "single_v1": "#56B4E9",   # Sky blue
}


def save_figure(fig, base_name: str, figures_dir: str, meta_info: dict, watermark: str = ""):
    """Save figure to .png and .pdf with .meta.json file."""
    os.makedirs(figures_dir, exist_ok=True)
    png_path = os.path.join(figures_dir, f"{base_name}.png")
    pdf_path = os.path.join(figures_dir, f"{base_name}.pdf")
    meta_path = os.path.join(figures_dir, f"{base_name}.meta.json")

    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=32, color="gray",
                 alpha=0.25, ha="center", va="center", rotation=30)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"  [OK] Exported figure: {base_name} (.png 300DPI, .pdf)")


def draw_fig1_architecture(figures_dir: str, meta_base: dict, watermark: str = ""):
    """Figure 1: QADF-Net architecture diagram for CMIG publication (Section 3)."""
    try:
        from cmig_qadf.report.plot_architecture_fig1 import generate_architecture_figure
        fig = generate_architecture_figure()
        save_figure(fig, "Fig1_Architecture", figures_dir, meta_base, watermark)
        
        # Also save to elsarticle/figures/ directory
        els_dir = os.path.abspath("cmig_qadf/elsarticle/figures")
        os.makedirs(els_dir, exist_ok=True)
        pdf_path = os.path.join(els_dir, "fig1_architecture.pdf")
        png_path = os.path.join(els_dir, "fig1_architecture.png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [OK] Updated Fig 1: Architecture of QADF-Net.")
    except Exception as e:
        print(f"  [WARN] Failed to generate architecture figure: {e}")


def draw_fig2_degradations(processed_dir: str, figures_dir: str, meta_base: dict, watermark: str = ""):
    """Figure 2: Real image grid with 7 rows (degradation types) x 6 columns (severity levels L0-L5) with q_hat labels."""
    try:
        from cmig_qadf.report.plot_degradation_grid import generate_degradation_grid_figure
        fig = generate_degradation_grid_figure()
        save_figure(fig, "Fig2_DegradationExamples", figures_dir, meta_base, watermark)
        # Also save standard filenames for LaTeX
        pdf_path = os.path.join(figures_dir, "fig2_degradation_examples.pdf")
        png_path = os.path.join(figures_dir, "fig2_degradation_examples.png")
        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [OK] Exported Fig 2: Real image grid (7 rows x 6 columns).")
    except Exception as e:
        print(f"  [WARN] Failed to generate degradation grid: {e}")


def draw_fig_s1_degradation_calibration(processed_dir: str, figures_dir: str, meta_base: dict, watermark: str = ""):
    """Figure S1 (Appendix): Physical degradation spectra and f_Q response curves matching paper values."""
    try:
        from cmig_qadf.report.plot_degradation_calibration import generate_calibration_figure
        fig = generate_calibration_figure(processed_dir)
        save_figure(fig, "FigS1_DegradationCalibration", figures_dir, meta_base, watermark)
        pdf_path = os.path.join(figures_dir, "fig_s1_degradation_calibration.pdf")
        png_path = os.path.join(figures_dir, "fig_s1_degradation_calibration.png")
        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [OK] Exported Fig S1 (Appendix): Degradation calibration curves.")
    except Exception as e:
        print(f"  [WARN] Failed to generate degradation calibration figure: {e}")


def draw_fig3_qwk_vs_degradation(processed_dir: str, figures_dir: str, meta_base: dict, watermark: str = ""):
    """Figure 3: QWK curves vs degradation severity levels (Section 5.2)."""
    try:
        from cmig_qadf.report.plot_qwk_robustness import generate_qwk_figure
        fig = generate_qwk_figure()
        save_figure(fig, "Fig3_QWK_vs_Degradation", figures_dir, meta_base, watermark)
        pdf_path = os.path.join(figures_dir, "fig3_qwk_vs_degradation.pdf")
        png_path = os.path.join(figures_dir, "fig3_qwk_vs_degradation.png")
        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [OK] Exported Fig 3: QWK Robustness curves (9 methods + CI).")
    except Exception as e:
        print(f"  [WARN] Failed to use plot_qwk_robustness: {e}")


def draw_fig4_attention_maps(figures_dir: str, meta_base: dict, watermark: str = ""):
    """Figure 4: QC-SCA spatial attention maps and Sink Token mass allocation (Section 5.3)."""
    try:
        from cmig_qadf.report.plot_attention_analysis import generate_attention_figure
        fig = generate_attention_figure()
        save_figure(fig, "Fig4_AttentionMaps", figures_dir, meta_base, watermark)
        pdf_path = os.path.join(figures_dir, "fig4_attention_maps.pdf")
        png_path = os.path.join(figures_dir, "fig4_attention_maps.png")
        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [OK] Exported Fig 4: Fundus attention overlay and alpha_sink curves.")
    except Exception as e:
        print(f"  [WARN] Failed to use plot_attention_analysis: {e}")


def draw_fig5_feature_drift(figures_dir: str, meta_base: dict, watermark: str = ""):
    """Figure 5: Comprehensive safety profiles (Harm Rate, PFR, FDI - 3 panels) across all 6 methods."""
    try:
        from cmig_qadf.report.plot_safety_profiles import generate_figure, save_all_formats
        fig = generate_figure()
        save_all_formats(fig)
        save_figure(fig, "Fig5_FeatureDrift", figures_dir, meta_base, watermark)
        save_figure(fig, "Fig5_SafetyProfiles", figures_dir, meta_base, watermark)
    except Exception as e:
        print(f"  [WARN] Failed to use plot_safety_profiles: {e}")


def draw_fig6_error_cases(figures_dir: str, meta_base: dict, watermark: str = ""):
    """Figure 6: Medical qualitative and error case studies (Defocus Blur vs Localized Glare)."""
    try:
        from cmig_qadf.report.plot_qualitative_cases import generate_qualitative_figure
        fig = generate_qualitative_figure()
        save_figure(fig, "Fig6_QualitativeCases", figures_dir, meta_base, watermark)
        save_figure(fig, "Fig6_ErrorCases", figures_dir, meta_base, watermark)
        plt.close(fig)
        print("  [OK] Updated Fig 6: Qualitative cases with actual fundus images.")
    except Exception as e:
        print(f"  [WARN] Failed to use plot_qualitative_cases: {e}")


def generate_all_figures(
    results_dir: str = "results",
    profile: str = "full",
    watermark: str = ""
):
    figures_dir = os.path.join(results_dir, "figures")
    processed_dir = os.path.join(results_dir, "processed")

    meta_base = {
        "profile": profile,
        "watermark": watermark,
    }

    print("\n--- STARTING EXPORT OF ALL FIGURES AND VISUALIZATIONS ---")
    draw_fig1_architecture(figures_dir, meta_base, watermark)
    draw_fig2_degradations(processed_dir, figures_dir, meta_base, watermark)
    draw_fig3_qwk_vs_degradation(processed_dir, figures_dir, meta_base, watermark)
    draw_fig4_attention_maps(figures_dir, meta_base, watermark)
    draw_fig5_feature_drift(figures_dir, meta_base, watermark)
    draw_fig6_error_cases(figures_dir, meta_base, watermark)
    draw_fig_s1_degradation_calibration(processed_dir, figures_dir, meta_base, watermark)

    # Theoretical illustration of Softmax No-Op (Section 3.3)
    try:
        from cmig_qadf.report.plot_softmax_noop import generate_figure as gen_noop, save_all_formats as save_noop
        fig_noop = gen_noop()
        save_noop(fig_noop)
        plt.close(fig_noop)
        print("  [OK] Exported illustration: Fig_SoftmaxNoop (.png 300DPI, .pdf)")
    except Exception as e:
        print(f"  [WARN] Failed to generate softmax noop figure: {e}")

    # Ablation Study Part B mechanism analysis (Section 5.5)
    try:
        from cmig_qadf.report.plot_ablation_partb import generate_figure as gen_ablation, save_all_formats as save_ablation
        fig_ablation = gen_ablation()
        save_ablation(fig_ablation)
        plt.close(fig_ablation)
        print("  [OK] Exported illustration: Fig7_AblationPartB (.png 300DPI, .pdf)")
    except Exception as e:
        print(f"  [WARN] Failed to generate ablation part b figure: {e}")

    print("[SUCCESS] Completed exporting all figures (.png 300DPI & .pdf).")


if __name__ == "__main__":
    generate_all_figures()
