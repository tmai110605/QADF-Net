"""
cmig_qadf/report/plot_architecture_fig1.py
==========================================
Figure 1: Architecture of QADF-Net for Elsevier CMIG manuscript.
Scientific medical design (Nature/IEEE TMI/CMIG):
- Stage 1: Dual-View Fundus Inputs (real DeepDRiD images), Shared Backbone (EfficientNet-B4), Token Grids, Frozen Quality Estimator (f_Q).
- Stage 2: Bidirectional Quality-Conditioned Spatial Cross-Attention (QC-SCA) with detailed Sink Token mechanism (tau_sink, v_sink=0),
           Additive Logit Bias (|beta|log q), Scale-V modulation (x q), and Quality-Weighted Consensus Pooling.
- Stage 3: CORAL Ordinal Classification Head with 4 monotonic thresholds (b_0 >= b_1 >= b_2 >= b_3) and DR grade predictions (ICDR 0-4).

Exports vector (.pdf) and high-resolution (.png 300 DPI) formats.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from PIL import Image, ImageDraw


def load_fundus_circular_crop(img_path: str, size: int = 320):
    """Load actual fundus image and apply soft circular crop."""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Create soft circular alpha mask
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((4, 4, size - 4, size - 4), fill=255)
        
        rgba = Image.new("RGBA", (size, size))
        rgba.paste(img, (0, 0), mask=mask)
        return np.array(rgba)
    except Exception as e:
        print(f"[WARN] Could not load fundus crop from {img_path}: {e}")
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        arr[:, :, 0] = 180
        arr[:, :, 1] = 60
        arr[:, :, 2] = 40
        arr[:, :, 3] = 255
        return arr


def generate_architecture_figure() -> plt.Figure:
    """Render complete QADF-Net architecture diagram for CMIG publication."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
    })

    fig = plt.figure(figsize=(17.2, 8.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    # Harmonious Clinical Palette
    C_BG_STAGE1 = "#F8FAFC"
    C_BORDER_STAGE1 = "#CBD5E1"
    C_HEADER_STAGE1 = "#0F172A"

    C_BG_STAGE2 = "#FFFDFD"
    C_BORDER_STAGE2 = "#FECACA"
    C_HEADER_STAGE2 = "#881337"

    C_BG_STAGE3 = "#F8FAF8"
    C_BORDER_STAGE3 = "#BBF7D0"
    C_HEADER_STAGE3 = "#14532D"

    C_BLUE = "#1D4ED8"
    C_TEAL = "#0F766E"
    C_VERMILLION = "#DC2626"
    C_AMBER = "#D97706"
    C_PURPLE = "#6B21A8"
    C_GREEN = "#15803D"
    C_SLATE = "#334155"

    def draw_card(x, y, w, h, bg_col, border_col, lw=1.2, radius=0.7, zorder=1):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius}",
            fc=bg_col, ec=border_col, lw=lw, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_header_badge(x, y, w, h, text, bg_col, text_col="white", fontsize=8.2):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35",
            fc=bg_col, ec="none", zorder=3
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight="bold", color=text_col, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color="#475569", lw=1.4, style="->", rad=0.0, zorder=5):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            connectionstyle=f"arc3,rad={rad}",
            color=color, lw=lw,
            mutation_scale=10, zorder=zorder
        )
        ax.add_patch(arrow)

    def draw_orthogonal_arrow(pts, color="#475569", lw=1.3, zorder=5):
        for i in range(len(pts) - 2):
            ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]], color=color, lw=lw, zorder=zorder)
        draw_arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, zorder=zorder)

    # =========================================================================
    # 1. THREE MAJOR STAGE CONTAINERS (BACKGROUND CONTAINERS)
    # =========================================================================
    # Stage 1 Card
    draw_card(1.2, 1.2, 31.0, 51.6, C_BG_STAGE1, C_BORDER_STAGE1, lw=1.2, radius=0.6, zorder=1)
    draw_header_badge(2.5, 50.4, 28.4, 2.0, "STAGE 1: DUAL-VIEW FEATURE EXTRACTION & QUALITY", C_HEADER_STAGE1, fontsize=8.0)

    # Stage 2 Card (Hero Innovation)
    draw_card(33.2, 1.2, 44.2, 51.6, C_BG_STAGE2, C_BORDER_STAGE2, lw=1.6, radius=0.6, zorder=1)
    draw_header_badge(34.5, 50.4, 41.6, 2.0, "STAGE 2: BIDIRECTIONAL QC-SCA & ADAPTIVE SINK FUSION (PROPOSED)", C_HEADER_STAGE2, fontsize=8.0)

    # Stage 3 Card
    draw_card(78.4, 1.2, 20.4, 51.6, C_BG_STAGE3, C_BORDER_STAGE3, lw=1.2, radius=0.6, zorder=1)
    draw_header_badge(79.6, 50.4, 18.0, 2.0, "STAGE 3: ORDINAL CORAL HEAD", C_HEADER_STAGE3, fontsize=8.0)

    # =========================================================================
    # STAGE 1: DUAL-VIEW INPUTS, BACKBONE, TOKENS, AND QUALITY ESTIMATOR
    # =========================================================================
    img_disc_path = "F:/DR/DeepDRiD/regular_fundus_images/Online-Challenge1&2-Evaluation/Images/347/347_l1.jpg"
    img_macula_path = "F:/DR/DeepDRiD/regular_fundus_images/Online-Challenge1&2-Evaluation/Images/347/347_l2.jpg"

    macula_arr = load_fundus_circular_crop(img_macula_path, size=240)
    disc_arr = load_fundus_circular_crop(img_disc_path, size=240)

    # View 1: Macula Image Box (Top Left)
    draw_card(2.4, 34.2, 8.8, 14.8, "white", "#93C5FD", lw=1.2, zorder=2)
    ax.imshow(macula_arr, extent=(2.8, 10.8, 38.0, 48.0), zorder=3)
    ax.text(6.8, 36.8, "View 1: Macula ($I_1$)", ha="center", va="center", fontsize=7.8, weight="bold", color=C_BLUE, zorder=4)
    ax.text(6.8, 35.3, "448 × 448 × 3 (RGB)", ha="center", va="center", fontsize=6.8, color="#64748B", zorder=4)

    # View 2: Optic Disc Image Box (Bottom Left)
    draw_card(2.4, 17.6, 8.8, 14.8, "white", "#99F6E4", lw=1.2, zorder=2)
    ax.imshow(disc_arr, extent=(2.8, 10.8, 21.4, 31.4), zorder=3)
    ax.text(6.8, 20.2, "View 2: Optic Disc ($I_2$)", ha="center", va="center", fontsize=7.8, weight="bold", color=C_TEAL, zorder=4)
    ax.text(6.8, 18.7, "448 × 448 × 3 (RGB)", ha="center", va="center", fontsize=6.8, color="#64748B", zorder=4)

    # Shared Backbone Box (EfficientNet-B4)
    draw_card(12.8, 20.5, 8.2, 25.5, "#FEF9C3", "#F59E0B", lw=1.4, zorder=2)
    ax.text(16.9, 44.0, "Shared Backbone", ha="center", va="center", fontsize=7.8, weight="bold", color="#92400E", zorder=4)
    ax.text(16.9, 42.4, "EfficientNet-B4", ha="center", va="center", fontsize=7.6, weight="bold", color="#78350F", zorder=4)
    ax.text(16.9, 40.8, "($C_b = 1792$)", ha="center", va="center", fontsize=6.8, color="#78350F", zorder=4)
    
    # Feature blocks representation
    for idx, (cy, col) in enumerate([(38.2, "#FDE047"), (36.2, "#FACC15"), (34.2, "#EAB308"), (32.2, "#CA8A04")]):
        ax.add_patch(patches.Rectangle((13.6, cy - 0.7), 6.6, 1.4, fc=col, ec="#A16207", lw=0.8, zorder=3))
        ax.text(16.9, cy, f"Stage {idx+2} Features", ha="center", va="center", fontsize=6.2, color="#451A03", zorder=4)

    ax.text(16.9, 29.5, "Shared Weights $\\theta_{\\mathrm{bb}}$", ha="center", va="center", fontsize=7.2, weight="bold", color="#92400E", zorder=4)
    ax.text(16.9, 28.0, "[Pretrained & Frozen]", ha="center", va="center", fontsize=6.5, style="italic", color="#78350F", zorder=4)
    ax.text(16.9, 25.5, "1×1 Conv + AdaptPool\n$d=128, G=14$", ha="center", va="center", fontsize=6.4, color="#451A03", zorder=4)

    # Arrows from Inputs into Backbone
    draw_arrow(11.2, 41.5, 12.8, 36.0, color=C_BLUE, lw=1.3)
    draw_arrow(11.2, 25.0, 12.8, 30.5, color=C_TEAL, lw=1.3)

    # Spatial Tokens Cards (Right side of Stage 1)
    # View 1 Tokens (Top)
    draw_card(22.6, 34.5, 8.8, 14.5, "#EFF6FF", "#BFDBFE", lw=1.2, zorder=2)
    ax.text(27.0, 47.4, "Spatial Tokens $F_1$", ha="center", va="center", fontsize=7.8, weight="bold", color=C_BLUE, zorder=4)
    ax.text(27.0, 45.8, "$N = 14 \\times 14 = 196$", ha="center", va="center", fontsize=6.8, color="#1E40AF", zorder=4)
    ax.text(27.0, 44.3, "Channel $d = 128$", ha="center", va="center", fontsize=6.8, color="#1E40AF", zorder=4)
    
    # 2D Grid representation View 1
    gx0, gy0 = 24.5, 37.0
    for r in range(4):
        for c in range(5):
            ax.add_patch(patches.Rectangle((gx0 + c*1.0, gy0 + (3-r)*1.0), 0.85, 0.85,
                                           fc="#DBEAFE" if (r+c)%2==0 else "#93C5FD", ec="#3B82F6", lw=0.6, zorder=3))
    ax.text(27.0, 35.5, "+ 2D Pos & View Embed", ha="center", va="center", fontsize=6.3, color="#1D4ED8", weight="bold", zorder=4)

    # View 2 Tokens (Bottom)
    draw_card(22.6, 17.6, 8.8, 14.5, "#F0FDFA", "#99F6E4", lw=1.2, zorder=2)
    ax.text(27.0, 30.5, "Spatial Tokens $F_2$", ha="center", va="center", fontsize=7.8, weight="bold", color=C_TEAL, zorder=4)
    ax.text(27.0, 28.9, "$N = 14 \\times 14 = 196$", ha="center", va="center", fontsize=6.8, color="#115E59", zorder=4)
    ax.text(27.0, 27.4, "Channel $d = 128$", ha="center", va="center", fontsize=6.8, color="#115E59", zorder=4)
    
    # 2D Grid representation View 2
    gx0, gy0 = 24.5, 20.1
    for r in range(4):
        for c in range(5):
            ax.add_patch(patches.Rectangle((gx0 + c*1.0, gy0 + (3-r)*1.0), 0.85, 0.85,
                                           fc="#CCFBF1" if (r+c)%2==0 else "#5EEAD4", ec="#0D9488", lw=0.6, zorder=3))
    ax.text(27.0, 18.6, "+ 2D Pos & View Embed", ha="center", va="center", fontsize=6.3, color="#0F766E", weight="bold", zorder=4)

    # Arrows from Backbone to Token Grids
    draw_arrow(21.0, 36.0, 22.6, 41.5, color=C_BLUE, lw=1.3)
    draw_arrow(21.0, 30.5, 22.6, 25.0, color=C_TEAL, lw=1.3)

    # Quality Estimator f_Q Card at Bottom of Stage 1
    draw_card(2.4, 2.4, 29.0, 13.5, "white", "#E9D5FF", lw=1.3, zorder=2)
    draw_header_badge(3.2, 13.5, 18.2, 1.8, "Quality Estimator $f_Q$ (MobileNetV3)", C_PURPLE, fontsize=7.2)
    ax.text(26.2, 14.4, "[Frozen $\\theta_Q$]", ha="center", va="center", fontsize=6.6, style="italic", color="#6B21A8", zorder=4)
    
    ax.text(16.9, 11.6, "Supervised by Clinical Clarity, Field Definition & Artefacts:\n$\\mathcal{L}_{\\mathrm{quality}} = \\mathcal{L}_{\\mathrm{MSE}}(\\hat{q}, q^{\\mathrm{human}}) + 0.5\\mathcal{L}_{\\mathrm{rank}}$",
            ha="center", va="center", fontsize=6.6, color="#475569", zorder=4)
    
    # Two Quality Output Gauges
    draw_card(3.8, 3.6, 12.5, 5.8, "#FAF5FF", "#C084FC", lw=1.0, zorder=3)
    ax.text(10.0, 7.6, "Macula Quality $\\hat{q}_1$", ha="center", va="center", fontsize=7.2, weight="bold", color="#6B21A8", zorder=4)
    ax.text(10.0, 5.8, "$q_1 = 0.912 \\in (0, 1]$", ha="center", va="center", fontsize=7.0, weight="bold", color="#1F2937", zorder=4)
    ax.text(10.0, 4.4, "High Clinical Fidelity", ha="center", va="center", fontsize=5.8, color="#059669", zorder=4)

    draw_card(17.5, 3.6, 12.5, 5.8, "#FAF5FF", "#C084FC", lw=1.0, zorder=3)
    ax.text(23.7, 7.6, "Optic Disc Quality $\\hat{q}_2$", ha="center", va="center", fontsize=7.2, weight="bold", color="#6B21A8", zorder=4)
    ax.text(23.7, 5.8, "$q_2 = 0.885 \\in (0, 1]$", ha="center", va="center", fontsize=7.0, weight="bold", color="#1F2937", zorder=4)
    ax.text(23.7, 4.4, "High Clinical Fidelity", ha="center", va="center", fontsize=5.8, color="#059669", zorder=4)

    # Arrows from inputs down to f_Q
    draw_orthogonal_arrow([(6.8, 17.6), (6.8, 15.5), (9.0, 15.5)], color=C_PURPLE, lw=1.1)

    # =========================================================================
    # STAGE 2: BIDIRECTIONAL QC-SCA & ADAPTIVE SINK FUSION (HERO SECTION)
    # =========================================================================
    # Detailed QC-SCA mechanism frame (Upper Box: Y = [20.0, 49.5])
    draw_card(34.2, 20.0, 42.2, 29.5, "white", "#FCA5A5", lw=1.5, zorder=2)
    draw_header_badge(35.0, 47.7, 25.5, 1.8, "QC-SCA Inner Mechanism ($F_1 \\leftarrow F_2$ Stream)", C_VERMILLION, fontsize=7.4)
    ax.text(68.5, 48.6, "Multi-Head Cross-Attention ($H=4, d_h=32$)", ha="center", va="center", fontsize=7.2, color="#991B1B", weight="bold", zorder=4)

    # Column 1: Projections Q1, K2, V2 (X = [35.2, 41.2])
    draw_card(35.2, 40.5, 6.2, 5.6, "#EFF6FF", "#93C5FD", lw=1.0, zorder=3)
    ax.text(38.3, 44.3, "Queries $Q_1$", ha="center", va="center", fontsize=7.2, weight="bold", color=C_BLUE, zorder=4)
    ax.text(38.3, 42.5, "$F_1 W_q$", ha="center", va="center", fontsize=6.8, color="#1E40AF", zorder=4)
    ax.text(38.3, 41.2, "$196 \\times 32$", ha="center", va="center", fontsize=6.0, color="#64748B", zorder=4)

    draw_card(35.2, 33.2, 6.2, 5.6, "#F0FDFA", "#99F6E4", lw=1.0, zorder=3)
    ax.text(38.3, 37.0, "Keys $K_2$", ha="center", va="center", fontsize=7.2, weight="bold", color=C_TEAL, zorder=4)
    ax.text(38.3, 35.2, "$F_2 W_k$", ha="center", va="center", fontsize=6.8, color="#115E59", zorder=4)
    ax.text(38.3, 33.9, "$196 \\times 32$", ha="center", va="center", fontsize=6.0, color="#64748B", zorder=4)

    draw_card(35.2, 21.5, 6.2, 5.6, "#F0FDFA", "#99F6E4", lw=1.0, zorder=3)
    ax.text(38.3, 25.3, "Values $V_2$", ha="center", va="center", fontsize=7.2, weight="bold", color=C_TEAL, zorder=4)
    ax.text(38.3, 23.5, "$F_2 W_v$", ha="center", va="center", fontsize=6.8, color="#115E59", zorder=4)
    ax.text(38.3, 22.2, "$196 \\times 32$", ha="center", va="center", fontsize=6.0, color="#64748B", zorder=4)

    # Scale-V Modulation Box on Value Path
    draw_card(43.0, 21.5, 7.8, 5.6, "#FEF3C7", "#FCD34D", lw=1.1, zorder=3)
    ax.text(46.9, 25.3, "Scale-V Modulation", ha="center", va="center", fontsize=6.8, weight="bold", color="#B45309", zorder=4)
    ax.text(46.9, 23.4, "$\\tilde{V}_2 = q_2 W_v \\mathbf{f}_2$", ha="center", va="center", fontsize=6.8, weight="bold", color="#92400E", zorder=4)
    ax.text(46.9, 22.2, "Linear attenuation", ha="center", va="center", fontsize=5.8, color="#78350F", zorder=4)
    draw_arrow(41.4, 24.3, 43.0, 24.3, color=C_TEAL, lw=1.2)

    # Learnable Sink Token Box (tau_sink) - Placed clearly above Scale-V, alongside Keys
    draw_card(43.0, 31.2, 9.0, 8.4, "#FEF3C7", "#D97706", lw=1.2, zorder=3)
    draw_header_badge(43.6, 37.8, 7.8, 1.5, r"Sink Token $\tau_{\mathrm{sink}}$", "#B45309", fontsize=6.5)
    ax.text(47.5, 36.0, r"Sink Key $k_{\mathrm{sink}} \in \mathbb{R}^{32}$", ha="center", va="center", fontsize=6.4, weight="bold", color="#78350F", zorder=4)
    ax.text(47.5, 34.3, "Sink Logit (Unbiased):", ha="center", va="center", fontsize=5.8, style="italic", color="#78350F", zorder=4)
    ax.text(47.5, 32.8, r"$s_{i,\mathrm{sink}} = \frac{Q_1^{(i)}k_{\mathrm{sink}}^\top}{\sqrt{d_h}}$", ha="center", va="center", fontsize=6.5, weight="bold", color="#92400E", zorder=4)

    # Zero Value Vector Callout Box
    draw_card(52.5, 21.5, 6.8, 5.6, "#FDE68A", "#B45309", lw=1.1, zorder=3)
    ax.text(55.9, 25.3, "Sink Value Vector", ha="center", va="center", fontsize=6.4, weight="bold", color="#78350F", zorder=4)
    ax.text(55.9, 23.4, r"$v_{\mathrm{sink}} \equiv \mathbf{0} \in \mathbb{R}^{32}$", ha="center", va="center", fontsize=7.2, weight="bold", color="#92400E", zorder=4)
    ax.text(55.9, 22.2, "Absorbs noise cleanly", ha="center", va="center", fontsize=5.6, color="#78350F", zorder=4)

    # Logits & Quality Additive Bias Box (X = [53.5, 63.8], Y = [31.5, 46.8])
    draw_card(53.5, 31.5, 10.5, 15.2, "#FEF2F2", "#F87171", lw=1.2, zorder=3)
    draw_header_badge(54.2, 44.8, 9.1, 1.6, "Affinity Logits", "#DC2626", fontsize=6.8)
    
    ax.text(58.7, 43.0, r"Dot Product ($j \leq N$):", ha="center", va="center", fontsize=6.2, color="#991B1B", zorder=4)
    ax.text(58.7, 41.5, r"$s_{ij}^0 = \frac{Q_1^{(i)}(K_2^{(j)})^\top}{\sqrt{d_h}}$", ha="center", va="center", fontsize=6.8, color="#7F1D1D", zorder=4)
    
    # Injected Additive Logit Bias
    draw_card(54.2, 33.5, 9.1, 6.8, "#FEE2E2", "#EF4444", lw=1.1, zorder=4)
    ax.text(58.7, 38.6, "Quality Logit Penalty:", ha="center", va="center", fontsize=6.2, weight="bold", color="#991B1B", zorder=5)
    ax.text(58.7, 36.8, r"$\tilde{s}_{ij} = s_{ij}^0 + |\beta|\log(q_2 + \epsilon)$", ha="center", va="center", fontsize=6.6, weight="bold", color="#7F1D1D", zorder=5)
    ax.text(58.7, 34.8, "Applied ONLY to Genuine Keys!\nSink logit $s_{i,\\mathrm{sink}}$ has NO penalty",
            ha="center", va="center", fontsize=5.4, color="#991B1B", zorder=5)

    # Normalized Softmax Routing Box (X = [65.2, 75.4], Y = [31.5, 46.8])
    draw_card(65.2, 31.5, 10.4, 15.2, "#F5F3FF", "#A78BFA", lw=1.2, zorder=3)
    draw_header_badge(65.9, 44.8, 9.0, 1.6, "Softmax Routing", C_PURPLE, fontsize=6.8)
    
    ax.text(70.4, 43.0, "Over $N+1$ Candidates:", ha="center", va="center", fontsize=6.2, color="#5B21B6", zorder=4)
    ax.text(70.4, 40.5, r"$\alpha_{ij} = \frac{\exp(\tilde{s}_{ij})}{\exp(s_{i,\mathrm{sink}}) + \sum_{k=1}^N \exp(\tilde{s}_{ik})}$",
            ha="center", va="center", fontsize=6.6, weight="bold", color="#4C1D95", zorder=4)
    
    # Degradation Behavior Callout
    draw_card(65.8, 32.5, 9.2, 6.5, "#EDE9FE", "#8B5CF6", lw=1.0, zorder=4)
    ax.text(70.4, 37.4, r"When $q_2 \to 0$ (Degraded):", ha="center", va="center", fontsize=6.0, weight="bold", color="#6D28D9", zorder=5)
    ax.text(70.4, 35.6, r"$\alpha_{i,\mathrm{sink}} \to 1, \quad \sum_{j=1}^N \alpha_{ij} \to 0$",
            ha="center", va="center", fontsize=6.6, weight="bold", color="#5B21B6", zorder=5)
    ax.text(70.4, 33.6, "Surplus mass drains safely to sink!", ha="center", va="center", fontsize=5.5, style="italic", color="#4C1D95", zorder=5)

    # Connections into Logits and Softmax
    draw_arrow(41.4, 42.5, 53.5, 42.5, color=C_BLUE, lw=1.2)
    draw_arrow(41.4, 36.0, 53.5, 40.0, color=C_TEAL, lw=1.2)
    draw_arrow(52.0, 32.8, 53.5, 33.5, color="#B45309", lw=1.2)
    draw_arrow(64.0, 39.0, 65.2, 39.0, color=C_VERMILLION, lw=1.2)

    # Delivered Message Box (Bottom Right of QC-SCA Detail: X = [60.5, 75.4], Y = [21.5, 29.8])
    draw_card(60.5, 21.5, 15.1, 8.5, "#ECFDF5", "#6EE7B7", lw=1.2, zorder=3)
    ax.text(68.0, 28.5, "Delivered Message $m_{1 \\leftarrow 2}$ & Certified Norm", ha="center", va="center", fontsize=6.8, weight="bold", color=C_GREEN, zorder=4)
    ax.text(68.0, 26.6, "$m_{1 \\leftarrow 2}^{(i)} = q_2 \\sum_{j=1}^N \\alpha_{ij} W_v \\mathbf{f}_2^{(j)} + \\alpha_{i,\\mathrm{sink}} \\mathbf{0} = q_2 g_i(q_2) \\mathbf{m}_i^{\\mathrm{std}}$",
            ha="center", va="center", fontsize=6.5, weight="bold", color="#065F46", zorder=4)
    ax.text(68.0, 24.3, "Lemma 1 Norm Bound (CMIG Theoretical Guarantee):", ha="center", va="center", fontsize=5.8, color="#047857", zorder=4)
    ax.text(68.0, 22.8, r"$\|m_{1 \leftarrow 2}^{(i)}\|_2 \leq q_2 (1 - \alpha_{i,\mathrm{sink}}) \max_j \|W_v \mathbf{f}_2^{(j)}\|_2$",
            ha="center", va="center", fontsize=6.4, weight="bold", color="#065F46", zorder=4)

    # Connect Value Path and Softmax into Message Box
    draw_arrow(50.8, 24.3, 52.5, 24.3, color=C_TEAL, lw=1.2)
    draw_arrow(59.3, 24.3, 60.5, 24.3, color="#B45309", lw=1.2)
    draw_arrow(70.4, 31.5, 70.4, 30.0, color=C_PURPLE, lw=1.2)

    # -------------------------------------------------------------------------
    # STAGE 2 LOWER REGION: UPDATED STREAMS & QUALITY-WEIGHTED CONSENSUS
    # -------------------------------------------------------------------------
    # Card 1: Dual Updated Feature Streams (X = [34.2, 53.0], Y = [2.4, 18.5])
    draw_card(34.2, 2.4, 18.8, 16.2, "white", "#93C5FD", lw=1.2, zorder=2)
    draw_header_badge(35.0, 16.8, 17.2, 1.6, "Bidirectional Updated Features", C_BLUE, fontsize=7.0)
    
    # Updated F1 Stream
    draw_card(35.0, 9.8, 17.2, 6.2, "#EFF6FF", "#BFDBFE", lw=1.0, zorder=3)
    ax.text(43.6, 14.5, "View 1 Updated: $\\hat{F}_1$", ha="center", va="center", fontsize=7.2, weight="bold", color=C_BLUE, zorder=4)
    ax.text(43.6, 12.8, "$\\hat{F}_1 = \\mathrm{LayerNorm}(F_1 + m_{1 \\leftarrow 2} W_o)$", ha="center", va="center", fontsize=6.6, weight="bold", color="#1E40AF", zorder=4)
    ax.text(43.6, 11.0, "Preserves primary macula context if disc degraded", ha="center", va="center", fontsize=5.6, style="italic", color="#2563EB", zorder=4)

    # Updated F2 Stream (Symmetric)
    draw_card(35.0, 3.2, 17.2, 6.0, "#F0FDFA", "#99F6E4", lw=1.0, zorder=3)
    ax.text(43.6, 7.8, "View 2 Updated: $\\hat{F}_2$ (Symmetric QC-SCA)", ha="center", va="center", fontsize=7.0, weight="bold", color=C_TEAL, zorder=4)
    ax.text(43.6, 6.2, "$\\hat{F}_2 = \\mathrm{LayerNorm}(F_2 + m_{2 \\leftarrow 1} W_o)$", ha="center", va="center", fontsize=6.6, weight="bold", color="#0F766E", zorder=4)
    ax.text(43.6, 4.6, "Conditioned on Macula Quality $q_1 \\in (0, 1]$", ha="center", va="center", fontsize=5.6, style="italic", color="#0D9488", zorder=4)

    # Connect Message m_{1 <- 2} to Updated F1
    draw_orthogonal_arrow([(68.0, 21.5), (68.0, 19.2), (43.6, 19.2), (43.6, 16.0)], color=C_GREEN, lw=1.3)

    # Card 2: Quality-Weighted Consensus Pooling (X = [54.2, 76.4], Y = [2.4, 18.5])
    draw_card(54.2, 2.4, 22.2, 16.2, "#FFFBEB", "#F59E0B", lw=1.3, zorder=2)
    draw_header_badge(55.0, 16.8, 20.6, 1.6, "Quality-Weighted Consensus Pooling", "#B45309", fontsize=7.2)

    ax.text(65.3, 14.8, "Global Average Pooling of Updated Tokens:", ha="center", va="center", fontsize=6.6, color="#78350F", zorder=4)
    ax.text(65.3, 13.2, "$\\mathbf{z}_1 = \\mathrm{GAP}(\\hat{F}_1) \\in \\mathbb{R}^{128}, \\quad \\mathbf{z}_2 = \\mathrm{GAP}(\\hat{F}_2) \\in \\mathbb{R}^{128}$",
            ha="center", va="center", fontsize=6.6, weight="bold", color="#92400E", zorder=4)

    ax.text(65.3, 11.2, "Dynamic Clinical Reliability Weights:", ha="center", va="center", fontsize=6.6, color="#78350F", zorder=4)
    ax.text(65.3, 9.6, "$w_1 = \\frac{q_1}{q_1 + q_2}, \\quad w_2 = \\frac{q_2}{q_1 + q_2}$",
            ha="center", va="center", fontsize=7.2, weight="bold", color="#B45309", zorder=4)

    # Unified Consensus Representation Box
    draw_card(55.2, 3.2, 20.2, 5.0, "#FEF3C7", "#D97706", lw=1.2, zorder=3)
    ax.text(65.3, 6.7, r"Fused Consensus Representation $\mathbf{f}_{\mathrm{cons}}$:", ha="center", va="center", fontsize=6.8, weight="bold", color="#78350F", zorder=4)
    ax.text(65.3, 4.8, r"$\mathbf{f}_{\mathrm{cons}} = w_1 \mathbf{z}_1 + w_2 \mathbf{z}_2 \in \mathbb{R}^{128}$",
            ha="center", va="center", fontsize=7.6, weight="bold", color="#92400E", zorder=4)

    # Connect F1, F2 to Consensus
    draw_arrow(52.2, 12.9, 54.2, 12.9, color=C_BLUE, lw=1.3)
    draw_arrow(52.2, 6.2, 54.2, 6.2, color=C_TEAL, lw=1.3)

    # Routing Quality Signals from f_Q into Stage 2
    draw_orthogonal_arrow([(30.0, 7.2), (32.6, 7.2), (32.6, 36.5), (35.0, 36.5)], color=C_PURPLE, lw=1.1)
    ax.text(32.6, 22.0, "Quality Signals\n$q_1, q_2 \\in (0, 1]$", ha="center", va="center", fontsize=6.0,
            color=C_PURPLE, weight="bold", rotation=90, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_PURPLE, lw=0.8))

    # Stage 1 to Stage 2 Token Connections
    draw_arrow(31.4, 41.5, 34.2, 41.5, color=C_BLUE, lw=1.3)
    draw_arrow(31.4, 25.0, 34.2, 25.0, color=C_TEAL, lw=1.3)

    # =========================================================================
    # STAGE 3: ORDINAL CORAL CLASSIFIER & CLINICAL DIAGNOSTIC PREDICTIONS
    # =========================================================================
    # Arrow from Consensus vector into Stage 3
    draw_orthogonal_arrow([(75.4, 6.0), (77.4, 6.0), (77.4, 26.0), (79.2, 26.0)], color="#B45309", lw=1.5)
    ax.text(77.4, 16.0, "$\\mathbf{f}_{\\mathrm{cons}}$", ha="center", va="center", fontsize=7.8, weight="bold", color="#B45309", zorder=5)

    # Projection to 4 Rank Tasks Card (X = [79.2, 97.8], Y = [24.0, 49.5])
    draw_card(79.2, 24.0, 18.6, 25.5, "white", "#86EFAC", lw=1.2, zorder=2)
    draw_header_badge(80.0, 47.7, 17.0, 1.8, "Rank-Ordered Binary Tasks ($K-1=4$)", C_GREEN, fontsize=7.2)
    
    ax.text(88.5, 45.8, "Shared Projection Vector: $\\mathbf{w} \\in \\mathbb{R}^{128}$", ha="center", va="center", fontsize=6.8, color="#14532D", zorder=4)
    ax.text(88.5, 44.0, r"Monotonic Thresholds: $b_0 \geq b_1 \geq b_2 \geq b_3$", ha="center", va="center", fontsize=6.8, weight="bold", color="#166534", zorder=4)
    
    task_info = [
        ("Task 0: $P(y > 0)$", "$b_0 = +2.42$", "$\\hat{p}_0 = 0.98$", "#DC2626", True),
        ("Task 1: $P(y > 1)$", "$b_1 = +1.15$", "$\\hat{p}_1 = 0.85$", "#EA580C", True),
        ("Task 2: $P(y > 2)$", "$b_2 = -0.74$", "$\\hat{p}_2 = 0.18$", "#4B5563", False),
        ("Task 3: $P(y > 3)$", "$b_3 = -2.10$", "$\\hat{p}_3 = 0.04$", "#4B5563", False),
    ]

    for idx, (t_name, b_val, p_val, col, active) in enumerate(task_info):
        ty = 38.6 - idx * 4.6
        bg_bar = "#DCFCE7" if active else "#F3F4F6"
        bd_bar = "#86EFAC" if active else "#D1D5DB"
        draw_card(80.2, ty, 16.6, 3.8, bg_bar, bd_bar, lw=0.9, zorder=3)
        ax.text(82.8, ty + 1.9, t_name, ha="center", va="center", fontsize=6.6, weight="bold", color="#1F2937", zorder=4)
        ax.text(87.2, ty + 1.9, b_val, ha="center", va="center", fontsize=6.2, color="#4B5563", zorder=4)
        ax.text(91.5, ty + 1.9, p_val, ha="center", va="center", fontsize=6.8, weight="bold", color=col, zorder=4)
        status_txt = "> 0.5 (Pass)" if active else "< 0.5 (Fail)"
        ax.text(94.8, ty + 1.9, status_txt, ha="center", va="center", fontsize=5.8, color="#059669" if active else "#6B7280", zorder=4)

    # Predicted Grade Equation & Result Card (X = [79.2, 97.8], Y = [2.4, 22.8])
    draw_card(79.2, 2.4, 18.6, 20.4, "#F0FDF4", "#4ADE80", lw=1.3, zorder=2)
    draw_header_badge(80.0, 20.8, 17.0, 1.8, "Predicted ICDR DR Grade $\\hat{y}$", "#15803D", fontsize=7.2)

    ax.text(88.5, 19.0, r"$\hat{y} = \sum_{r=0}^3 \mathbb{I}(P(y > r) > 0.5) \in \{0, 1, 2, 3, 4\}$",
            ha="center", va="center", fontsize=6.8, weight="bold", color="#14532D", zorder=4)
    ax.text(88.5, 17.4, "Binary Sum: $1 + 1 + 0 + 0 = 2$", ha="center", va="center", fontsize=7.2, weight="bold", color="#15803D", zorder=4)

    # Clinical Severity Meter
    dr_grades = [
        ("0", "No DR", "#E5E7EB", "#4B5563", False),
        ("1", "Mild", "#FED7AA", "#C2410C", False),
        ("2", "Moderate", "#FDE047", "#854D0E", True),
        ("3", "Severe", "#FECACA", "#B91C1C", False),
        ("4", "PDR", "#FCA5A5", "#991B1B", False),
    ]

    for g_idx, (g_num, g_lbl, g_bg, g_tc, is_pred) in enumerate(dr_grades):
        bx = 80.2 + g_idx * 3.32
        by = 8.6
        bw = 3.1
        bh = 4.6
        border_c = "#E11D48" if is_pred else "#D1D5DB"
        lw_box = 1.6 if is_pred else 0.8
        draw_card(bx, by, bw, bh, g_bg, border_c, lw=lw_box, radius=0.2, zorder=3)
        ax.text(bx + bw/2, by + 3.0, f"Grade {g_num}", ha="center", va="center", fontsize=6.0, weight="bold", color=g_tc, zorder=4)
        ax.text(bx + bw/2, by + 1.4, g_lbl, ha="center", va="center", fontsize=5.4, color=g_tc, zorder=4)
        if is_pred:
            # Highlight border & badge for predicted grade
            ax.annotate("PREDICTED", xy=(bx + bw/2, by + bh), xytext=(bx + bw/2, by + bh + 1.4),
                        ha="center", va="bottom", fontsize=6.0, weight="bold", color="#E11D48",
                        arrowprops=dict(arrowstyle="->", color="#E11D48", lw=1.3), zorder=5)

    ax.text(88.5, 5.0, "Rank-consistent ordinal probabilities guarantee\nno catastrophic multi-grade diagnostic inversion",
            ha="center", va="center", fontsize=6.0, style="italic", color="#166534", zorder=4)

    return fig


def save_all_formats(fig: plt.Figure):
    """Save figure to both results/figures/ and cmig_qadf/elsarticle/figures/."""
    targets = [
        os.path.abspath("cmig_qadf/elsarticle/figures/fig1_architecture.pdf"),
        os.path.abspath("cmig_qadf/elsarticle/figures/fig1_architecture.png"),
        os.path.abspath("results/figures/Fig1_Architecture.pdf"),
        os.path.abspath("results/figures/Fig1_Architecture.png"),
    ]

    for path in targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fmt = "pdf" if path.endswith(".pdf") else "png"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {path}")


if __name__ == "__main__":
    print("\n--- GENERATING FIGURE 1: ARCHITECTURE OF QADF-NET ---")
    fig = generate_architecture_figure()
    save_all_formats(fig)
    plt.close(fig)
    print("--- FIGURE 1 COMPLETE ---\n")
