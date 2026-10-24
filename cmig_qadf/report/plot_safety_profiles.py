"""
plot_safety_profiles.py
=======================
Generates Figure 5: 3 panels comprising Harm Rate (%), Prediction Flip Rate (PFR, %), 
and Feature Drift Index (FDI) across 6 degradation levels (L0 to L5) for all 6 fusion methods:
  - Concat
  - Average
  - Max-Pooling
  - Cross-Attention
  - Co-Attention
  - QADF-Net (Ours)

Values match Table 4 (tab:harm_flip) in manuscript.
Includes 95% Confidence Interval (CI) based on 5 seeds.
Publication standards: sans-serif font >= 8pt, colorblind-friendly Wong palette,
vector PDF and 300 DPI PNG formats.
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
    "figure.autolayout": False,
})

# 6 severity levels: L0 (clean), L1, L2, L3, L4, L5
levels = np.array([0, 1, 2, 3, 4, 5])
level_labels = ["L0\n(clean)", "L1", "L2", "L3", "L4", "L5"]

# Multiplier for 95% CI with n=5 (t_df=4, p=0.025: t = 2.776 / sqrt(5) ≈ 1.2415)
CI_MULT = 2.776 / np.sqrt(5)

# Mean values matching Table 4 (tab:harm_flip)
METHODS = [
    {
        "name": "QADF-Net (Ours)",
        "color": "#D55E00",      # Vermillion highlight
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 10,
        "alpha_fill": 0.20,
        "harm_mean": np.array([12.30, 12.59, 13.10, 13.96, 14.29, 14.49]),
        "harm_std":  np.array([2.50, 1.65, 1.40, 1.29, 1.38, 1.13]),
        "pfr_mean":  np.array([0.00, 13.97, 17.94, 20.89, 22.20, 21.84]),
        "pfr_std":   np.array([0.00, 1.75, 2.54, 2.58, 3.11, 3.64]),
        "fdi_mean":  np.array([0.0000, 0.2747, 0.3707, 0.4256, 0.4686, 0.4793]),
        "fdi_std":   np.array([0.0000, 0.0182, 0.0329, 0.0433, 0.0515, 0.0551]),
    },
    {
        "name": "Co-Attention",
        "color": "#0072B2",      # Blue
        "marker": "s",
        "linestyle": "-",
        "linewidth": 1.5,
        "zorder": 6,
        "alpha_fill": 0.12,
        "harm_mean": np.array([9.00, 9.84, 12.14, 15.51, 19.56, 21.50]),
        "harm_std":  np.array([1.76, 1.87, 1.97, 2.04, 3.24, 3.03]),
        "pfr_mean":  np.array([0.00, 9.34, 16.07, 21.80, 27.46, 30.50]),
        "pfr_std":   np.array([0.00, 0.61, 2.16, 2.32, 4.04, 4.12]),
        "fdi_mean":  np.array([0.0000, 0.2572, 0.4220, 0.5976, 0.7846, 0.9531]),
        "fdi_std":   np.array([0.0000, 0.0220, 0.0410, 0.0600, 0.0800, 0.0980]),
    },
    {
        "name": "Cross-Attention",
        "color": "#009E73",      # Green
        "marker": "^",
        "linestyle": "--",
        "linewidth": 1.5,
        "zorder": 7,
        "alpha_fill": 0.12,
        "harm_mean": np.array([13.00, 13.41, 15.64, 17.39, 18.69, 18.79]),
        "harm_std":  np.array([3.36, 3.33, 3.60, 2.33, 2.65, 2.83]),
        "pfr_mean":  np.array([0.00, 12.13, 19.04, 23.29, 26.79, 28.67]),
        "pfr_std":   np.array([0.00, 0.50, 0.98, 1.72, 1.76, 3.21]),
        "fdi_mean":  np.array([0.0000, 0.2635, 0.3777, 0.4641, 0.5471, 0.6141]),
        "fdi_std":   np.array([0.0000, 0.0348, 0.0467, 0.0550, 0.0591, 0.0597]),
    },
    {
        "name": "Max-Pooling",
        "color": "#56B4E9",      # Sky blue
        "marker": "D",
        "linestyle": "-.",
        "linewidth": 1.4,
        "zorder": 5,
        "alpha_fill": 0.10,
        "harm_mean": np.array([13.00, 12.71, 14.36, 16.70, 19.96, 23.74]),
        "harm_std":  np.array([1.10, 1.13, 1.56, 1.86, 3.20, 4.44]),
        "pfr_mean":  np.array([0.00, 12.09, 19.09, 25.24, 30.46, 35.79]),
        "pfr_std":   np.array([0.00, 1.08, 1.57, 1.46, 2.74, 3.67]),
        "fdi_mean":  np.array([0.0000, 0.2862, 0.4279, 0.5753, 0.7395, 0.8988]),
        "fdi_std":   np.array([0.0000, 0.0250, 0.0380, 0.0580, 0.0750, 0.0920]),
    },
    {
        "name": "Average",
        "color": "#E69F00",      # Yellow-Orange
        "marker": "v",
        "linestyle": ":",
        "linewidth": 1.5,
        "zorder": 4,
        "alpha_fill": 0.10,
        "harm_mean": np.array([11.60, 12.66, 15.96, 19.41, 23.51, 27.23]),
        "harm_std":  np.array([2.99, 2.36, 3.46, 3.99, 6.37, 7.78]),
        "pfr_mean":  np.array([0.00, 11.76, 19.09, 26.69, 33.47, 39.19]),
        "pfr_std":   np.array([0.00, 0.81, 1.49, 3.59, 5.39, 6.36]),
        "fdi_mean":  np.array([0.0000, 0.2301, 0.3574, 0.4943, 0.6453, 0.7924]),
        "fdi_std":   np.array([0.0000, 0.0305, 0.0440, 0.0653, 0.0851, 0.1050]),
    },
    {
        "name": "Concat",
        "color": "#CC79A7",      # Reddish purple
        "marker": "x",
        "linestyle": ":",
        "linewidth": 1.5,
        "zorder": 3,
        "alpha_fill": 0.10,
        "harm_mean": np.array([12.50, 12.99, 15.57, 18.29, 24.24, 28.79]),
        "harm_std":  np.array([2.59, 1.52, 0.52, 0.67, 1.32, 1.33]),
        "pfr_mean":  np.array([0.00, 12.14, 19.66, 25.76, 34.57, 41.61]),
        "pfr_std":   np.array([0.00, 1.13, 0.93, 0.83, 1.71, 1.31]),
        "fdi_mean":  np.array([0.0000, 0.2528, 0.4012, 0.5751, 0.7728, 0.9663]),
        "fdi_std":   np.array([0.0000, 0.0132, 0.0092, 0.0192, 0.0564, 0.0900]),
    },
]


def generate_figure():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharex=True)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.82, wspace=0.25)

    # -------------------------------------------------------------
    # Panel (a): Harm Rate (%)
    # -------------------------------------------------------------
    ax_a = axes[0]
    for m in METHODS:
        y = m["harm_mean"]
        ci = CI_MULT * m["harm_std"]
        ax_a.plot(levels, y, marker=m["marker"], color=m["color"],
                  linestyle=m["linestyle"], linewidth=m["linewidth"],
                  label=m["name"], zorder=m["zorder"])
        ax_a.fill_between(levels, y - ci, y + ci, color=m["color"],
                          alpha=m["alpha_fill"], zorder=m["zorder"] - 1)

    ax_a.set_title("(a) Harm Rate (%) vs. Severity", weight="bold")
    ax_a.set_ylabel("Harm Rate (% of pairs)")
    ax_a.set_ylim(6.0, 32.0)
    ax_a.set_xticks(levels)
    ax_a.set_xticklabels(level_labels)
    ax_a.grid(True, linestyle="--", alpha=0.45)

    # -------------------------------------------------------------
    # Panel (b): Prediction Flip Rate (PFR, %)
    # -------------------------------------------------------------
    ax_b = axes[1]
    for m in METHODS:
        y = m["pfr_mean"]
        ci = CI_MULT * m["pfr_std"]
        # Ensure CI is non-negative at L0
        low = np.maximum(0.0, y - ci)
        high = y + ci
        ax_b.plot(levels, y, marker=m["marker"], color=m["color"],
                  linestyle=m["linestyle"], linewidth=m["linewidth"],
                  label=m["name"], zorder=m["zorder"])
        ax_b.fill_between(levels, low, high, color=m["color"],
                          alpha=m["alpha_fill"], zorder=m["zorder"] - 1)

    ax_b.set_title("(b) Prediction Flip Rate (PFR, %)", weight="bold")
    ax_b.set_ylabel("Flip Rate (% relative to L0)")
    ax_b.set_ylim(-1.0, 46.0)
    ax_b.set_xticks(levels)
    ax_b.set_xticklabels(level_labels)
    ax_b.grid(True, linestyle="--", alpha=0.45)

    # -------------------------------------------------------------
    # Panel (c): Feature Drift Index (FDI)
    # -------------------------------------------------------------
    ax_c = axes[2]
    for m in METHODS:
        y = m["fdi_mean"]
        ci = CI_MULT * m["fdi_std"]
        low = np.maximum(0.0, y - ci)
        high = y + ci
        ax_c.plot(levels, y, marker=m["marker"], color=m["color"],
                  linestyle=m["linestyle"], linewidth=m["linewidth"],
                  label=m["name"], zorder=m["zorder"])
        ax_c.fill_between(levels, low, high, color=m["color"],
                          alpha=m["alpha_fill"], zorder=m["zorder"] - 1)

    ax_c.set_title("(c) Feature Drift Index (FDI)", weight="bold")
    ax_c.set_ylabel(r"FDI: $\|\mathbf{f}_{\mathrm{cons}}^{(L)} - \mathbf{f}_{\mathrm{cons}}^{(0)}\|_2 / \|\mathbf{f}_{\mathrm{cons}}^{(0)}\|_2$")
    ax_c.set_ylim(-0.02, 1.10)
    ax_c.set_xticks(levels)
    ax_c.set_xticklabels(level_labels)
    ax_c.grid(True, linestyle="--", alpha=0.45)

    # Common x-axis label
    for ax in axes:
        ax.set_xlabel("Degradation Severity Level (View 2 degraded)")

    # Common legend placed above the 3 panels
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.98),
               ncol=6, frameon=True, fancybox=True, framealpha=0.95,
               edgecolor="#CCCCCC", fontsize=8.2, columnspacing=1.2)

    return fig


def save_all_formats(fig):
    targets = [
        # 1. In elsarticle/figures (used for LaTeX compilation)
        os.path.join("cmig_qadf", "elsarticle", "figures", "fig5_safety_profiles.pdf"),
        os.path.join("cmig_qadf", "elsarticle", "figures", "fig5_safety_profiles.png"),
        os.path.join("cmig_qadf", "elsarticle", "figures", "fig5_feature_drift.pdf"),
        os.path.join("cmig_qadf", "elsarticle", "figures", "fig5_feature_drift.png"),
        # 2. In results/figures (repository)
        os.path.join("results", "figures", "Fig5_SafetyProfiles.pdf"),
        os.path.join("results", "figures", "Fig5_SafetyProfiles.png"),
        os.path.join("results", "figures", "Fig5_FeatureDrift.pdf"),
        os.path.join("results", "figures", "Fig5_FeatureDrift.png"),
    ]

    for p in targets:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if p.endswith(".pdf"):
            fig.savefig(p, format="pdf", bbox_inches="tight")
        else:
            fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {p}")

    # Export meta json file
    meta_path = os.path.join("results", "figures", "Fig5_SafetyProfiles.meta.json")
    meta = {
        "title": "Figure 5: Degradation Safety Profiles (Harm Rate, PFR, FDI)",
        "panels": ["(a) Harm Rate (%)", "(b) Prediction Flip Rate (PFR, %)", "(c) Feature Drift Index (FDI)"],
        "methods": [m["name"] for m in METHODS],
        "n_seeds": 5,
        "ci_type": "95% Student-t CI across 5 seeds (mean over 7 degradations)",
        "source": "Table 4 (tab:harm_flip) and results/processed/s2_harm_empirical.json"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] Saved meta: {meta_path}")


if __name__ == "__main__":
    fig = generate_figure()
    save_all_formats(fig)
    plt.close(fig)
    print("\n[SUCCESS] Completed generating Fig 5 (3 panels) across 6 methods with 95% CI.")
