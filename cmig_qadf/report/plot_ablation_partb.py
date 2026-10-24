"""
plot_ablation_partb.py
======================
Generates Figure for Ablation Study Section 5.5, Table 7 Part B:
  - Panel (a): Delta AUDC vs. Plain Cross-Attention (None) with error bars (+/- 1 SD).
  - Panel (b): Delta Harm Rate at L5 (pp reduction, higher reduction is safer) with error bars (+/- 1 SD).
  - Panel (c): 2x2 Factorial Interaction plot: Logit Bias x Sink Token,
               demonstrating super-additive interaction and Softmax No-Op.

Aesthetic alignment with Elsevier / CMIG standards:
  - Standard font: DejaVu Sans, aligned with Fig 5, Fig 6, Fig Softmax No-op.
  - Minimalist palette: Neutral text (#202124).
  - Intentional emphasis: Blue (#1A73E8) for Proposed / With Sink, Slate (#5F6368) for baselines.
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

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Elsevier / CMIG aesthetic configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#202124",
    "axes.labelsize": 8.0,
    "axes.labelcolor": "#202124",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": "#202124",
    "ytick.color": "#202124",
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.5,
    "text.color": "#202124",
    "mathtext.fontset": "dejavusans",
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
    "axes.edgecolor": "#DADCE0",
    "axes.linewidth": 0.8,
})

# Table 7 Part B data from official manuscript
# None (Baseline): AUDC = 0.8454 +/- 0.0127, HR_L5 = 18.72 +/- 2.41%
baseline = {
    "name": "None",
    "audc_mean": 0.8454,
    "audc_std": 0.0127,
    "hr_mean": 18.72,
    "hr_std": 2.41,
}

variants = [
    {
        "name": "Bias w/o Sink",
        "label": "Bias w/o Sink (No-Op)",
        "desc": "Logit bias only",
        "audc_mean": 0.8456,
        "audc_std": 0.0033,
        "hr_mean": 18.69,
        "hr_std": 2.37,
        "color": "#80868B",
        "is_proposed": False,
    },
    {
        "name": "Sink w/o Bias",
        "label": "Sink w/o Bias",
        "desc": "Sink token, fixed q=1",
        "audc_mean": 0.8478,
        "audc_std": 0.0084,
        "hr_mean": 18.15,
        "hr_std": 1.72,
        "color": "#80868B",
        "is_proposed": False,
    },
    {
        "name": "Scale-K only",
        "label": "Scale-K only",
        "desc": "Key scaling sqrt(q)",
        "audc_mean": 0.8493,
        "audc_std": 0.0092,
        "hr_mean": 17.92,
        "hr_std": 1.83,
        "color": "#80868B",
        "is_proposed": False,
    },
    {
        "name": "Bias + Sink only",
        "label": "Bias + Sink only",
        "desc": "Logit bias + sink (no Scale-V)",
        "audc_mean": 0.8564,
        "audc_std": 0.0056,
        "hr_mean": 16.38,
        "hr_std": 1.42,
        "color": "#5F6368",
        "is_proposed": False,
    },
    {
        "name": "Scale-V only",
        "label": "Scale-V only",
        "desc": "Value scaling q*V",
        "audc_mean": 0.8577,
        "audc_std": 0.0063,
        "hr_mean": 16.18,
        "hr_std": 1.36,
        "color": "#5F6368",
        "is_proposed": False,
    },
    {
        "name": "Q-Weighted GAP",
        "label": "Q-Weighted GAP (No Attn)",
        "desc": "Attention-free control",
        "audc_mean": 0.8562,
        "audc_std": 0.0058,
        "hr_mean": 15.68,
        "hr_std": 1.15,
        "color": "#673AB7",
        "is_proposed": False,
    },
    {
        "name": "QC-SCA (uniform)",
        "label": "QC-SCA only (uniform)",
        "desc": "Bias + Sink + Scale-V",
        "audc_mean": 0.8598,
        "audc_std": 0.0069,
        "hr_mean": 15.42,
        "hr_std": 1.12,
        "color": "#1976D2",
        "is_proposed": False,
    },
    {
        "name": "All (proposed)",
        "label": "All (Proposed QC-SCA)",
        "desc": "Full QC-SCA + consensus",
        "audc_mean": 0.8630,
        "audc_std": 0.0081,
        "hr_mean": 14.52,
        "hr_std": 0.89,
        "color": "#1A73E8",
        "is_proposed": True,
    },
]

def generate_figure():
    fig = plt.figure(figsize=(12.0, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1.05], wspace=0.34)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    y_pos = np.arange(len(variants))
    labels = [v["label"] for v in variants]

    # -------------------------------------------------------------
    # Panel (a): Delta AUDC vs None (+/- 1 SD)
    # -------------------------------------------------------------
    delta_audc = [v["audc_mean"] - baseline["audc_mean"] for v in variants]
    audc_err = [v["audc_std"] for v in variants]
    bar_colors_a = ["#1A73E8" if v["is_proposed"] else ("#7E57C2" if "GAP" in v["name"] else "#5F6368") for v in variants]

    bars_a = ax_a.barh(y_pos, [d * 100 for d in delta_audc], xerr=[e * 100 for e in audc_err],
                       height=0.56, color=bar_colors_a, alpha=0.88,
                       error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#3C4043"})

    ax_a.axvline(0, color="#202124", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels, fontsize=7.5)
    ax_a.set_xlabel(r"$\Delta\mathrm{AUDC} \times 10^{-2}$ vs. Plain Cross-Attn", fontsize=8.0)
    ax_a.set_title(r"(a) Robustness Gain ($\Delta\mathrm{AUDC}$ vs. None)", fontsize=8.5, fontweight="bold", pad=8)
    ax_a.set_xlim(-0.6, 3.2)
    ax_a.grid(True, axis="x", linestyle=":", alpha=0.5, color="#DADCE0")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Annotate values cleanly to the right of error bars to avoid collisions
    for i, (d, e, v) in enumerate(zip(delta_audc, audc_err, variants)):
        val_str = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
        font_weight = "bold" if v["is_proposed"] else "normal"
        text_color = "#1A73E8" if v["is_proposed"] else "#202124"
        # Place label past the error bar cap
        x_pos = max(0, d * 100) + e * 100 + 0.10
        ax_a.text(x_pos, i, val_str, va="center", ha="left", fontsize=7.0,
                  fontweight=font_weight, color=text_color)

    # -------------------------------------------------------------
    # Panel (b): Harm Rate Reduction (-Delta HR_L5, pp) vs None
    # -------------------------------------------------------------
    delta_hr = [baseline["hr_mean"] - v["hr_mean"] for v in variants]
    hr_err = [v["hr_std"] for v in variants]
    bar_colors_b = ["#1A73E8" if v["is_proposed"] else ("#7E57C2" if "GAP" in v["name"] else "#5F6368") for v in variants]

    bars_b = ax_b.barh(y_pos, delta_hr, xerr=hr_err,
                       height=0.56, color=bar_colors_b, alpha=0.88,
                       error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#3C4043"})

    ax_b.axvline(0, color="#202124", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([])  # Align with ax_a
    ax_b.tick_params(left=False)
    ax_b.set_xlabel(r"Harm Rate Reduction $-\Delta\mathrm{HR}_{L5}$ (pp, higher is safer)", fontsize=8.0)
    ax_b.set_title(r"(b) Severe Harm Reduction ($L=5$)", fontsize=8.5, fontweight="bold", pad=8)
    ax_b.set_xlim(-1.2, 6.2)
    ax_b.grid(True, axis="x", linestyle=":", alpha=0.5, color="#DADCE0")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Annotate values cleanly to the right of error bars to avoid collisions
    for i, (d, e, v) in enumerate(zip(delta_hr, hr_err, variants)):
        val_str = f"+{d:.2f} pp" if d >= 0 else f"{d:.2f} pp"
        font_weight = "bold" if v["is_proposed"] else "normal"
        text_color = "#1A73E8" if v["is_proposed"] else "#202124"
        x_pos = max(0, d) + e + 0.18
        ax_b.text(x_pos, i, val_str, va="center", ha="left", fontsize=7.0,
                  fontweight=font_weight, color=text_color)

    # -------------------------------------------------------------
    # Panel (c): 2x2 Factorial Interaction (Logit Bias x Sink Token)
    # -------------------------------------------------------------
    x_base = np.array([0.0, 1.0])
    x_labels = ["Absent\n(Standard logit)", "Present\n($+\\beta\\log q_2$)"]

    # Slight dodge to avoid error bar collision at x=0
    dodge = 0.035
    x_no_sink = x_base - dodge
    x_sink = x_base + dodge

    # Without Sink Token: None (0.8454 +/- 0.0127) -> Bias w/o Sink (0.8456 +/- 0.0033)
    y_no_sink = [0.8454, 0.8456]
    err_no_sink = [0.0127, 0.0033]
    # With Sink Token: Sink w/o Bias (0.8478 +/- 0.0084) -> Bias + Sink (0.8564 +/- 0.0056)
    y_sink = [0.8478, 0.8564]
    err_sink = [0.0084, 0.0056]

    line1 = ax_c.errorbar(x_no_sink, y_no_sink, yerr=err_no_sink, fmt="s--", color="#5F6368",
                          linewidth=1.6, markersize=5.5, capsize=3.0, label="Without Sink Token (Softmax No-Op)",
                          ecolor="#5F6368")
    line2 = ax_c.errorbar(x_sink, y_sink, yerr=err_sink, fmt="o-", color="#1A73E8",
                          linewidth=1.8, markersize=6.0, capsize=3.0, label="With Sink Token (QC-SCA)",
                          ecolor="#1A73E8")

    ax_c.set_xticks(x_base)
    ax_c.set_xticklabels(x_labels, fontsize=7.5)
    ax_c.set_xlim(-0.25, 1.25)
    ax_c.set_ylim(0.838, 0.866)
    ax_c.set_ylabel(r"AUDC (mean $\pm$ SD)", fontsize=8.0)
    ax_c.set_title(r"(c) $2\times 2$ Interaction: Bias $\times$ Sink", fontsize=8.5, fontweight="bold", pad=8)
    ax_c.grid(True, linestyle=":", alpha=0.5, color="#DADCE0")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Slope annotations
    slope_no_sink = y_no_sink[1] - y_no_sink[0]  # +0.0002
    slope_sink = y_sink[1] - y_sink[0]        # +0.0086
    ax_c.text(0.5, 0.8437, rf"$\Delta = +{slope_no_sink:.4f}$ (Flat)", ha="center", va="top",
              fontsize=7.0, color="#5F6368", style="italic")
    ax_c.text(0.5, 0.8533, rf"$\Delta = +{slope_sink:.4f}$ (Sharp rise)", ha="center", va="bottom",
              fontsize=7.0, color="#1A73E8", fontweight="bold")

    # Super-additive box
    box_props = dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#DADCE0", alpha=0.95)
    synergy_text = (
        r"$\mathbf{Super\text{-}additive\ synergy\!:}$" + "\n"
        r"Joint gain: $\mathbf{+0.0110}$" + "\n"
        r"Sum of separate: $+0.0026$" + "\n"
        r"Interaction: $\mathbf{\Delta_{\mathrm{inter}} = +0.0084}$"
    )
    ax_c.text(0.04, 0.96, synergy_text, transform=ax_c.transAxes,
              fontsize=6.8, va="top", ha="left", bbox=box_props, color="#202124")

    ax_c.legend(loc="lower right", fontsize=7.0, frameon=True, facecolor="#FFFFFF",
                edgecolor="#DADCE0", framealpha=0.9)

    return fig

def save_all_formats(fig, output_dirs=None):
    if output_dirs is None:
        output_dirs = [
            "f:/DR/cmig_qadf/elsarticle/figures",
            "f:/DR/results/figures",
        ]
    base_name = "fig7_ablation_partb"
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        pdf_path = os.path.join(d, f"{base_name}.pdf")
        png_path = os.path.join(d, f"{base_name}.png")
        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {pdf_path}")
        print(f"  ✓ Saved: {png_path}")

if __name__ == "__main__":
    fig = generate_figure()
    save_all_formats(fig)
    print("✓ Successfully generated Fig 7 Ablation Part B!")
