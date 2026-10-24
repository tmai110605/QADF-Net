"""
plot_softmax_noop.py
====================
Generates illustration explaining the Softmax No-Op phenomenon (Section 3.3, following Remark 1).
Analytical / Formula-based toy demonstration:
  - Panel (a): Without Sink (Standard Softmax with logit bias beta log q):
               Attention bars at q = 1.0, 0.5, 0.1 are strictly identical (Softmax No-Op).
  - Panel (b): With Sink Token (QC-SCA with zero-value sink token tau_sink):
               As q decreases (1.0 -> 0.5 -> 0.1), attention mass smoothly concentrates on the sink token.
  - Panel (c): Total attention mass placed on genuine tokens vs q in [0.01, 1.0]:
               Constant 100% flatline vs Smooth, monotonic withdrawal curve.

Export formats: Vector PDF and 300 DPI PNG, styled for CMIG / Elsevier.
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
from matplotlib.patches import Patch

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
})

# Toy simulation parameters
# 4 genuine tokens k_1, k_2, k_3, k_4
s0 = np.array([2.5, 1.8, 1.0, 0.2])
tokens_labels = [r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$"]
s_sink = 0.0  # Logit of sink token (unbiased by log q)
beta = 1.6    # Scaling factor beta

qs = [1.0, 0.5, 0.1]
q_colors = {
    1.0: "#0072B2",  # Blue (Clean)
    0.5: "#E69F00",  # Amber/Yellow-Orange (Moderate)
    0.1: "#D55E00",  # Vermillion (Severe)
}
q_labels = {
    1.0: r"$q = 1.0$ (Clean)",
    0.5: r"$q = 0.5$ (Moderate)",
    0.1: r"$q = 0.1$ (Severe)",
}


def compute_weights():
    # 1. Without sink: alpha_j = exp(s0_j + beta log q) / sum(exp(s0_k + beta log q))
    # Because beta log q cancels: alpha_j = exp(s0_j) / sum(exp(s0_k))
    alpha_no_sink = {}
    for q in qs:
        s = s0 + beta * np.log(q)
        alpha_no_sink[q] = np.exp(s) / np.sum(np.exp(s))

    # 2. With sink:
    # Genuine tokens receive bias beta log q, sink token does not
    alpha_with_sink_gen = {}
    alpha_with_sink_val = {}
    for q in qs:
        s_gen = s0 + beta * np.log(q)
        all_s = np.append(s_gen, s_sink)
        weights = np.exp(all_s) / np.sum(np.exp(all_s))
        alpha_with_sink_gen[q] = weights[:-1]
        alpha_with_sink_val[q] = weights[-1]

    return alpha_no_sink, alpha_with_sink_gen, alpha_with_sink_val


def generate_figure():
    alpha_no_sink, alpha_with_sink_gen, alpha_with_sink_val = compute_weights()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.82, wspace=0.28)

    # Bar width and x positions
    x_gen = np.arange(len(s0))
    bar_width = 0.24

    # -------------------------------------------------------------
    # Panel (a): Without Sink Token (Softmax No-Op)
    # -------------------------------------------------------------
    ax_a = axes[0]
    for idx, q in enumerate(qs):
        offset = (idx - 1) * bar_width
        weights = alpha_no_sink[q] * 100  # Percentage
        ax_a.bar(x_gen + offset, weights, width=bar_width,
                 color=q_colors[q], label=q_labels[q],
                 edgecolor="#333333", linewidth=0.8, alpha=0.9)

    # Horizontal dotted guide indicating constant value
    for i in range(len(s0)):
        h = alpha_no_sink[1.0][i] * 100
        ax_a.plot([x_gen[i] - 1.3 * bar_width, x_gen[i] + 1.3 * bar_width], [h, h],
                  color="#5F6368", linestyle=":", linewidth=1.2)

    ax_a.set_title("(a) Without Sink Token: Softmax No-Op", weight="bold", color="#202124")
    ax_a.set_xlabel("Genuine Key Tokens")
    ax_a.set_ylabel("Attention Weight (%)")
    ax_a.set_xticks(x_gen)
    ax_a.set_xticklabels(tokens_labels)
    ax_a.set_ylim(0, 68)
    ax_a.grid(True, linestyle="--", alpha=0.45, axis="y")

    # Softmax Invariance callout card
    ax_a.text(0.50, 0.88, r"$\alpha_j(1.0) \equiv \alpha_j(0.5) \equiv \alpha_j(0.1)$" + "\n" +
              r"Softmax cancels $\beta\log q$" + "\n" + r"(Total genuine mass $\equiv 100\%$)",
              transform=ax_a.transAxes, ha="center", va="top",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DADCE0", alpha=0.95),
              fontsize=7.5, color="#202124")

    # -------------------------------------------------------------
    # Panel (b): With Sink Token (QC-SCA Mechanism)
    # -------------------------------------------------------------
    ax_b = axes[1]
    x_all = np.arange(len(s0) + 1)
    all_labels = tokens_labels + [r"$\tau_{\mathrm{sink}}$" + "\n" + r"$(v=\mathbf{0})$"]

    for idx, q in enumerate(qs):
        offset = (idx - 1) * bar_width
        w_gen = alpha_with_sink_gen[q] * 100
        w_sink = alpha_with_sink_val[q] * 100
        w_combined = np.append(w_gen, w_sink)

        # Genuine tokens bars
        ax_b.bar(x_all[:-1] + offset, w_combined[:-1], width=bar_width,
                 color=q_colors[q], edgecolor="#333333", linewidth=0.8, alpha=0.9)
        # Sink token bar with hatched pattern
        ax_b.bar(x_all[-1:] + offset, w_combined[-1:], width=bar_width,
                 color=q_colors[q], edgecolor="#333333", linewidth=0.8,
                 hatch="//", alpha=0.85)

    ax_b.set_title("(b) With Sink Token: QC-SCA Withdrawal", weight="bold", color="#202124")
    ax_b.set_xlabel("Key Tokens (Genuine Tokens + Sink)")
    ax_b.set_ylabel("Attention Weight (%)")
    ax_b.set_xticks(x_all)
    ax_b.set_xticklabels(all_labels)
    ax_b.set_ylim(0, 75)
    ax_b.grid(True, linestyle="--", alpha=0.45, axis="y")

    # Annotation: mass concentrates on sink
    ax_b.annotate("Mass concentrates\nonto sink token\n(up to 64.2%)",
                  xy=(4.0 + bar_width, 64.5), xytext=(2.3, 56),
                  arrowprops=dict(arrowstyle="->", color="#5F6368", lw=1.2),
                  fontsize=7.5, color="#202124", weight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#DADCE0", alpha=0.95))

    # -------------------------------------------------------------
    # Panel (c): Total Genuine Attention Mass vs Quality q
    # -------------------------------------------------------------
    ax_c = axes[2]
    q_cont = np.linspace(0.01, 1.0, 200)
    sum_exp_s0 = np.sum(np.exp(s0))
    exp_s_sink = np.exp(s_sink)

    # Without sink: total mass on genuine tokens = 1.0 (flat 100%)
    mass_no_sink = np.ones_like(q_cont) * 100.0

    # With sink: total mass on genuine tokens
    mass_with_sink = (q_cont**beta * sum_exp_s0) / (exp_s_sink + q_cont**beta * sum_exp_s0) * 100.0
    sink_mass_cont = exp_s_sink / (exp_s_sink + q_cont**beta * sum_exp_s0) * 100.0

    ax_c.plot(q_cont, mass_no_sink, color="#555555", linestyle="--", linewidth=1.8,
              label=r"Without Sink: $\sum \alpha_j \equiv 100\%$ (Forced)")
    ax_c.plot(q_cont, mass_with_sink, color="#D55E00", linestyle="-", linewidth=2.4,
              label=r"With Sink: Genuine Mass $\sum_{j=1}^N \alpha_{ij}$")
    ax_c.plot(q_cont, sink_mass_cont, color="#0072B2", linestyle="-.", linewidth=1.8,
              label=r"With Sink: Sink Mass $\alpha_{i,\mathrm{sink}}$")

    # Mark 3 points q = 1.0, 0.5, 0.1
    for q_pt in qs:
        m_val = (q_pt**beta * sum_exp_s0) / (exp_s_sink + q_pt**beta * sum_exp_s0) * 100.0
        ax_c.plot(q_pt, m_val, marker="o", color=q_colors[q_pt], markersize=6, zorder=5)

    ax_c.set_title("(c) Attention Mass vs. Source Quality $q$", weight="bold", color="#202124")
    ax_c.set_xlabel(r"Source View Quality Score $q \in (0, 1]$")
    ax_c.set_ylabel("Total Attention Mass (%)")
    ax_c.set_xlim(0.0, 1.02)
    ax_c.set_ylim(-2, 105)
    ax_c.grid(True, linestyle="--", alpha=0.45)
    ax_c.legend(loc="center left", fontsize=7.5, framealpha=0.92)

    # Shared legend above panels (a) and (b)
    handles, labels = ax_a.get_legend_handles_labels()
    # Add patch for sink token
    sink_patch = Patch(facecolor="#D55E00", edgecolor="#333333", hatch="//", label=r"Sink Token $\tau_{\mathrm{sink}}$ ($v=\mathbf{0}$)")
    handles.append(sink_patch)
    labels.append(r"Sink Token $\tau_{\mathrm{sink}}$ ($v=\mathbf{0}$)")

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.98),
               ncol=4, frameon=True, fancybox=True, framealpha=0.95,
               edgecolor="#DADCE0", fontsize=7.5, columnspacing=1.5)

    return fig


def save_all_formats(fig):
    targets = [
        # 1. In elsarticle/figures (for LaTeX compilation)
        os.path.join("cmig_qadf", "elsarticle", "figures", "fig_softmax_noop.pdf"),
        os.path.join("cmig_qadf", "elsarticle", "figures", "fig_softmax_noop.png"),
        # 2. In results/figures (repository)
        os.path.join("results", "figures", "Fig_SoftmaxNoop.pdf"),
        os.path.join("results", "figures", "Fig_SoftmaxNoop.png"),
    ]

    for p in targets:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if p.endswith(".pdf"):
            fig.savefig(p, format="pdf", bbox_inches="tight")
        else:
            fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {p}")

    meta_path = os.path.join("results", "figures", "Fig_SoftmaxNoop.meta.json")
    meta = {
        "title": "Illustration of the Softmax No-Op Phenomenon and QC-SCA Sink Mechanism",
        "description": "Toy analytical demonstration showing how additive logit bias cancels under softmax shift-invariance (a) and how a zero-value sink token enables smooth, monotonic attention mass withdrawal (b, c).",
        "parameters": {
            "genuine_logits_s0": s0.tolist(),
            "sink_logit_s_sink": s_sink,
            "beta": beta,
            "quality_levels_evaluated": qs
        },
        "nature": "Illustrative / Analytical formula-based (Remark 1, Section 3.3)"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] Saved meta: {meta_path}")


if __name__ == "__main__":
    fig = generate_figure()
    save_all_formats(fig)
    plt.close(fig)
    print("\n[SUCCESS] Completed generating Softmax No-Op figure (.PDF & .PNG 300DPI).")
