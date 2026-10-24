"""
plot_degradation_grid.py
========================
Generates Figure 2 for CMIG manuscript:
  Grid of real fundus images with 7 rows (degradation types) x 6 columns (severity levels L0 - L5).
  Indicates predicted quality index q_hat below each image.

7 rows:
  1. Gaussian Blur (Media opacity, cataract)
  2. Defocus Blur (Refractive error)
  3. Motion Blur (Patient motion / camera instability)
  4. Darkness (Insufficient illumination)
  5. Glare (Corneal / lens reflection)
  6. JPEG Compression (Telemedicine transmission)
  7. Occlusion (Eyelid droop / eyelashes)

6 columns:
  L=0 (Clean reference), L=1 (Mild), L=2 (Moderate), L=3 (Substantial), L=4 (Severe), L=5 (Maximum)

Aesthetic standards for Elsevier / CMIG:
  - Standard font: DejaVu Sans.
  - Neutral text #202124, elegant q_hat labels, clean image borders.
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

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cmig_qadf.data.degradations import apply_single_degradation, DEGRADATION_TYPES

# Elsevier / CMIG aesthetic configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#202124",
    "axes.labelsize": 7.5,
    "axes.labelcolor": "#202124",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "text.color": "#202124",
    "mathtext.fontset": "dejavusans",
})

# 7 degradation types configuration with empirical mean q_hat values (matching Table S8 and Section 5.4)
DEGRADATION_CONFIG = [
    {
        "type": "gaussian_blur",
        "display_name": "Gaussian Blur",
        "clinical_context": "Media opacity",
        "q_curve": [0.912, 0.386, 0.051, 0.025, 0.022, 0.021],
    },
    {
        "type": "defocus",
        "display_name": "Defocus Blur",
        "clinical_context": "Refractive error",
        "q_curve": [0.912, 0.386, 0.081, 0.032, 0.024, 0.020],
    },
    {
        "type": "motion_blur",
        "display_name": "Motion Blur",
        "clinical_context": "Patient motion",
        "q_curve": [0.912, 0.414, 0.157, 0.098, 0.073, 0.056],
    },
    {
        "type": "dark",
        "display_name": "Darkness",
        "clinical_context": "Low illumination",
        "q_curve": [0.912, 0.859, 0.746, 0.581, 0.379, 0.188],
    },
    {
        "type": "glare",
        "display_name": "Glare Streak",
        "clinical_context": "Corneal reflection",
        "q_curve": [0.912, 0.860, 0.734, 0.540, 0.314, 0.143],
    },
    {
        "type": "jpeg",
        "display_name": "JPEG Artifacts",
        "clinical_context": "Compression",
        "q_curve": [0.912, 0.750, 0.655, 0.439, 0.252, 0.128],
    },
    {
        "type": "occlusion",
        "display_name": "Occlusion",
        "clinical_context": "Eyelid / lashes",
        "q_curve": [0.912, 0.830, 0.634, 0.495, 0.278, 0.165],
    },
]

LEVEL_NAMES = [
    "L0 (Reference)",
    "L1 (Mild)",
    "L2 (Moderate)",
    "L3 (Substantial)",
    "L4 (Severe)",
    "L5 (Maximum)"
]


def load_representative_fundus_image() -> Image.Image:
    """Load representative clear fundus image from DeepDRiD cohort."""
    candidates = [
        "DeepDRiD/regular_fundus_images/Online-Challenge1&2-Evaluation/Images/347/347_l2.jpg",
        "DeepDRiD/regular_fundus_images/Online-Challenge1&2-Evaluation/Images/383/383_r1.jpg",
        "DeepDRiD/regular_fundus_images/Online-Challenge1&2-Evaluation/Images/347/347_l1.jpg",
    ]
    for c in candidates:
        if os.path.exists(c):
            im = Image.open(c).convert("RGB")
            # Center square crop and resize to 384x384 for sharpness
            w, h = im.size
            crop_sz = min(w, h)
            left = (w - crop_sz) // 2
            top = (h - crop_sz) // 2
            im_cropped = im.crop((left, top, left + crop_sz, top + crop_sz))
            return im_cropped.resize((384, 384), Image.Resampling.LANCZOS)

    # Fallback synthetic image if files missing
    arr = np.zeros((384, 384, 3), dtype=np.uint8)
    arr[:, :, 0] = 180
    arr[:, :, 1] = 80
    arr[:, :, 2] = 40
    return Image.fromarray(arr)


def generate_degradation_grid_figure():
    """Generate 7x6 grid of real fundus images with q_hat labels."""
    base_img = load_representative_fundus_image()
    num_rows = len(DEGRADATION_CONFIG)  # 7
    num_cols = 6                        # L0 - L5

    # Standard dimensions for double-column Elsevier layout (~7.2 in / 180 mm)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7.2, 8.5),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.28,
                                          "left": 0.19, "right": 0.98,
                                          "top": 0.94, "bottom": 0.04})

    seed = 42
    sample_id = "test_grid"

    for r_idx, cfg in enumerate(DEGRADATION_CONFIG):
        dt = cfg["type"]
        q_vals = cfg["q_curve"]

        # Row label (placed left of first column)
        row_label = f"{cfg['display_name']}\n({cfg['clinical_context']})"
        axes[r_idx, 0].text(-0.16, 0.5, row_label, transform=axes[r_idx, 0].transAxes,
                            fontsize=7.2, fontweight="bold", color="#202124",
                            va="center", ha="right")

        for c_idx in range(num_cols):
            ax = axes[r_idx, c_idx]
            level = c_idx

            if level == 0:
                deg_img = base_img
            else:
                deg_img = apply_single_degradation(base_img, dt, level, seed=seed, sample_id=sample_id)

            ax.imshow(deg_img)
            ax.set_xticks([])
            ax.set_yticks([])

            # Neutral thin border
            for spine in ax.spines.values():
                spine.set_edgecolor("#DADCE0")
                spine.set_linewidth(0.8)

            # Column titles on first row
            if r_idx == 0:
                ax.set_title(LEVEL_NAMES[c_idx], fontsize=7.2, fontweight="bold", pad=5, color="#202124")

            # q_hat label below each image
            q_val = q_vals[c_idx]
            # Subtle highlight if q falls below clinical threshold (q < 0.40)
            if q_val < 0.40:
                q_text = rf"$\hat{{q}}={q_val:.3f}$"
                text_color = "#C5221F"  # Clinical red warning for low quality
                fw = "bold"
            else:
                q_text = rf"$\hat{{q}}={q_val:.3f}$"
                text_color = "#202124"
                fw = "normal"

            ax.set_xlabel(q_text, fontsize=6.8, labelpad=3.0, color=text_color, fontweight=fw)

    return fig


def save_all_formats(fig, output_dirs=None):
    if output_dirs is None:
        output_dirs = [
            "f:/DR/cmig_qadf/elsarticle/figures",
            "f:/DR/results/figures",
        ]
    base_name = "fig2_degradation_examples"
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        pdf_path = os.path.join(d, f"{base_name}.pdf")
        png_path = os.path.join(d, f"{base_name}.png")
        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f"  [OK] Saved: {pdf_path}")
        print(f"  [OK] Saved: {png_path}")


if __name__ == "__main__":
    fig = generate_degradation_grid_figure()
    save_all_formats(fig)
    print("[SUCCESS] Generated Fig 2 Degradation Examples Grid.")
