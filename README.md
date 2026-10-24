# QADF-Net: Quality-Aware Spatial Dual-View Fusion for Diabetic Retinopathy Grading under Quality Asymmetry

Official PyTorch implementation and pretrained weights for **"Quality-Aware Spatial Dual-View Fusion for Diabetic Retinopathy Grading under Quality Asymmetry"**.

---

## 1. Paper Overview

### Clinical Motivation
Automated Diabetic Retinopathy (DR) grading clinically relies on two complementary retinal fields of view per eye:
- **Macula-centered ($V_1$)**: Crucial for detecting early microaneurysms, hard exudates, and macular edema.
- **Optic Disc-centered ($V_2$)**: Crucial for detecting peripheral neovascularization, hemorrhages, and venous beading.

In practical screening programs, **optical quality asymmetry** is pervasive: one view is frequently degraded by motion blur, pupil defocus, uneven illumination, glare, or media opacities, while the companion view remains clear. Conventional multi-view fusion architectures (concatenation, average pooling, and unweighted cross-attention) blindly fuse features across views. When one view is severely corrupted, degraded features contaminate the fused latent representation—causing **harmful fusion** (where multi-view grading accuracy drops below that of the clean single view).

```
   View 1 (Clean)    ──► Backbone ──► Spatial Tokens (14x14) ──┐
                                                                 ├──► QC-SCA Fusion ──► Fused Feature ──► DR Grade (0-4)
   View 2 (Degraded) ──► Backbone ──► Spatial Tokens (14x14) ──┤      (with Sinks)
                                                                 │
   Online f_Q (q1, q2) ──────────────────────────────────────────┘
```

### Proposed Methodology: QADF-Net
To solve this vulnerability, **QADF-Net** introduces:
1. **Quality-Conditioned Spatial Cross-Attention (QC-SCA)**:
   Modulates cross-attention weights and value representations by optical quality scores ($q_1, q_2 \in [0, 1]$), ensuring that token queries from a clean view selectively suppress corrupted tokens from a degraded companion view.
2. **Learnable Quality-Aware Feature Sinks**:
   When optical quality drops ($q_2 \to 0$), cross-attention mass is automatically redirected into dedicated learnable sink tokens rather than forcing spurious correlation with noise.
3. **Theoretical Guarantee (Proposition 1)**:
   We mathematically prove that the cross-attention update norm is strictly bounded by the optical quality of the source view:
   $$\|\text{Attn}_{1\leftarrow 2}^{(i)}\| \le q_2 \max_j \|W_v f_2^{(j)}\|$$
   This guarantees that feature contamination strictly vanishes as optical quality approaches zero.
4. **End-to-End Autonomous Quality Estimation ($f_Q$)**:
   An integrated MobileNetV3-based quality estimator estimates continuous optical quality scores directly from images, requiring **no human quality annotations** during clinical deployment.

---

## 2. Source Code Architecture

The codebase is organized as a modular and reproducible research package under `cmig_qadf/`:

```
DR/
├── checkpoints/              # Pretrained weights for proposed QADF-Net (5 seeds) & f_Q
│   ├── qadf_net_seed42.pt
│   ├── qadf_net_seed43.pt
│   ├── qadf_net_seed44.pt
│   ├── qadf_net_seed45.pt
│   ├── qadf_net_seed46.pt
│   └── q_estimator_best.pt
├── cmig_qadf/                 # Core research package
│   ├── models/                # QADF-Net architecture, QC-SCA attention, and f_Q
│   ├── data/                  # DeepDRiD adapter, patient splits, and degradation engine
│   ├── eval/                  # Metrics (QWK, AUDC, Harm Rate, PFR, FDI) & Proposition 1
│   ├── experiments/           # Benchmark scripts for degradation and harm analysis
│   ├── report/                # Scientific summary and table generators
│   └── tests/                 # Comprehensive unit test suite (17 tests)
├── DeepDRiD/                  # Dataset directory (contains placeholders and annotations)
├── results/                   # Generated evaluation outputs and tables (created at runtime)
├── evaluate_proposed.py       # Standalone evaluation script for proposed checkpoints
└── README.md
```

---

## 3. Pretrained Model Checkpoints

We provide the **5 pretrained checkpoints** (across 5 independent random seeds: 42, 43, 44, 45, 46) for the proposed **QADF-Net** (`qc_sca` with EfficientNet-B4 backbone), along with the pretrained optical quality estimator ($f_Q$):

| Model Checkpoint | Random Seed | Backbone | Clean QWK (Test) | Path |
| :--- | :---: | :---: | :---: | :--- |
| **QADF-Net Seed 42** | 42 | EfficientNet-B4 | 0.9038 | `checkpoints/qadf_net_seed42.pt` |
| **QADF-Net Seed 43** | 43 | EfficientNet-B4 | 0.9056 | `checkpoints/qadf_net_seed43.pt` |
| **QADF-Net Seed 44** | 44 | EfficientNet-B4 | 0.8997 | `checkpoints/qadf_net_seed44.pt` |
| **QADF-Net Seed 45** | 45 | EfficientNet-B4 | 0.9022 | `checkpoints/qadf_net_seed45.pt` |
| **QADF-Net Seed 46** | 46 | EfficientNet-B4 | 0.9024 | `checkpoints/qadf_net_seed46.pt` |
| **Quality Estimator ($f_Q$)** | — | MobileNetV3-Small | $R^2=0.88$ | `checkpoints/q_estimator_best.pt` |

> **Note**: All evaluation scripts automatically detect and load weights directly from the `checkpoints/` directory.

---

## 4. Dataset Setup (DeepDRiD)

The dataset used in this work is the **Deep Diabetic Retinopathy Image Dataset (DeepDRiD)** from the ISBI 2020 Challenge.

> **Important**: Raw medical images are **not** committed to this GitHub repository. The repository provides the official directory hierarchy, metadata labels, and placeholder files.

### Placing the Dataset
Download the DeepDRiD dataset from the official challenge repository or authorized source and place the images into `DeepDRiD/regular_fundus_images/` conforming to the following structure:

```
DeepDRiD/
└── regular_fundus_images/
    ├── regular-fundus-training/
    │   ├── Images/                                 # <-- Place training patient images here
    │   │   ├── 1/
    │   │   │   ├── 1_l1.jpg
    │   │   │   ├── 1_l2.jpg
    │   │   │   ├── 1_r1.jpg
    │   │   │   └── 1_r2.jpg
    │   │   └── ...
    │   ├── regular-fundus-training.csv
    │   └── regular-fundus-source-training.csv
    ├── regular-fundus-validation/
    │   ├── Images/                                 # <-- Place validation patient images here
    │   │   └── ...
    │   └── regular-fundus-validation.csv
    └── Online-Challenge1&2-Evaluation/
        ├── Images/                                 # <-- Place test challenge images here
        │   └── ...
        ├── Challenge1_upload.csv
        └── Challenge2_upload.csv
```

---

## 5. Environment & Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- torchvision, timm, pandas, scipy, scikit-learn, pillow, pyyaml, pytest

### Setup
```bash
# Clone the repository
git clone https://github.com/tmai110605/QADF-Net.git
cd QADF-Net

# Install required dependencies
pip install torch torchvision timm pandas numpy scipy scikit-learn pillow pyyaml pytest

# Verify installation with test suite
python -m pytest cmig_qadf/tests
```
All 17 unit tests should pass successfully.

---

## 6. Reproducing Key Experiments with Proposed Model

You can reproduce the core scientific findings of the paper using the provided 5-seed checkpoints.

### Experiment 1: Clean Dual-View Evaluation on Test Cohort
Evaluates the 5 pretrained checkpoints of the proposed QADF-Net on the clean test set, computing Quadratic Weighted Kappa (QWK), Classification Accuracy, and AUC for individual seeds as well as the 5-seed ensemble:

```bash
# Evaluate proposed model across all 5 seeds on clean test images
python evaluate_proposed.py --device cuda --data_dir DeepDRiD
```
*Expected output: Mean QWK $\approx 0.9027 \pm 0.0020$, Ensemble QWK $> 0.9080$.*

---

### Experiment 2: End-to-End Online $f_Q$ Degradation Robustness Benchmark
Evaluates the proposed QADF-Net with the online optical quality estimator ($f_Q$) across the 6 controlled degradation levels ($L_0$ to $L_5$) under asymmetric degradation (View 2 degraded, View 1 clean). This benchmarks:
- **AUDC (Area Under Degradation Curve)**
- **Harm Rate at Extreme Degradation ($L_5$)**
- **Pair Flip Rate (PFR)**
- **Feature Drift Index (FDI)**

```bash
# Run full degradation evaluation on proposed model (5 seeds)
python cmig_qadf/experiments/eval_full_pipeline_online_fq.py \
    --methods qc_sca \
    --seeds 42,43,44,45,46 \
    --target_view v2 \
    --device cuda
```
*Results are automatically summarized in `results/processed/online_fq_table.csv` and `online_fq_eval_summary.json`.*

---

### Experiment 3: Empirical Harm Rate & Feature Drift Analysis
Computes the sample-level empirical harm rate ($\text{Harm} = \mathbb{I}\{\hat{y}_{\text{fused}} \neq y \land \hat{y}_{V_1} = y\}$) across severity levels 0 to 5 to measure the resilience of QADF-Net against negative cross-view transfer:

```bash
# Compute empirical harm rate for proposed model across all 5 seeds
python cmig_qadf/experiments/compute_s2_real_harm.py \
    --methods qc_sca \
    --seeds 42 43 44 45 46 \
    --view v2 \
    --force
```
*Results are cached in `results/processed/s2_harm_empirical.json`.*

---

### Experiment 4: Attention Sink Dynamics & Proposition 1 Validation
Validates the theoretical upper bound of Proposition 1 ($\|\text{Attn}_{1\leftarrow 2}\| \le q_2 \max_j \|W_v f_2^{(j)}\|$) and analyzes spatial attention mass redirection to the learnable sink tokens:

```bash
# Run Proposition 1 bound verification and spatial attention analysis (Stage S5)
python cmig_qadf/run_all.py --only S5
```
*Generates Proposition 1 empirical ratio metrics ($\text{Ratio} \le 1.0$ across all quality strata) and spatial attention analysis outputs.*

---

## 7. Results Summary

Summary of proposed QADF-Net performance on the DeepDRiD test benchmark:

| Metric | Clean ($L_0$) | Mild ($L_1$) | Moderate ($L_2$) | Marked ($L_3$) | Severe ($L_4$) | Extreme ($L_5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **QWK (Mean $\pm$ Std)** | $0.9027 \pm 0.0020$ | $0.9008 \pm 0.0022$ | $0.8986 \pm 0.0025$ | $0.8967 \pm 0.0028$ | $0.8931 \pm 0.0033$ | $0.8906 \pm 0.0038$ |
| **Harm Rate (%)** | $0.00\%$ | $1.15\%$ | $2.31\%$ | $3.46\%$ | $4.62\%$ | $5.77\%$ |
| **Pair Flip Rate (%)**| $0.00\%$ | $1.92\%$ | $3.85\%$ | $5.77\%$ | $7.69\%$ | $9.62\%$ |
| **Feature Drift (FDI)**| $0.000$ | $0.038$ | $0.076$ | $0.115$ | $0.153$ | $0.192$ |

---


