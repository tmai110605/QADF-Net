"""
cmig_qadf/data/degradations.py
==============================
Controlled optical degradation suite across 7 types × 5 severity levels (Level 0 = clean, 1-5 increasing).
Fully deterministic based on SHA-256 hash seeding (seed, sample_id, deg_type, level, view).
Applicable to: View 1, View 2, or both.
"""

import io
import math
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional


DEGRADATION_TYPES = [
    "gaussian_blur",
    "defocus",
    "motion_blur",
    "dark",
    "glare",
    "jpeg",
    "occlusion",
]


def _get_rng(seed: int, sample_id: str, deg_type: str, level: int, view: str) -> np.random.RandomState:
    """Create deterministic pseudo-random number generator from SHA-256 hash."""
    key = f"{seed}_{sample_id}_{deg_type}_{level}_{view}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    int_seed = int(h[:8], 16)
    return np.random.RandomState(int_seed)


def apply_gaussian_blur(img: Image.Image, level: int, rng: np.random.RandomState) -> Image.Image:
    if level <= 0: return img
    radii = [0.0, 1.2, 2.5, 4.2, 6.5, 9.0]
    r = radii[min(level, 5)]
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def apply_defocus(img: Image.Image, level: int, rng: np.random.RandomState) -> Image.Image:
    if level <= 0: return img
    radii = [0, 2, 4, 7, 10, 14]
    r = radii[min(level, 5)]
    # Construct disk blur kernel
    size = 2 * r + 1
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0 / np.sum(mask)

    img_np = np.array(img, dtype=np.uint8)
    out_np = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(out_np)


def apply_motion_blur(img: Image.Image, level: int, rng: np.random.RandomState) -> Image.Image:
    if level <= 0: return img
    lengths = [0, 5, 11, 19, 29, 41]
    length = lengths[min(level, 5)]
    # Fixed 45-degree angle following evaluation protocol to guarantee monotonicity
    angle = 45.0

    # Linear motion blur kernel rotated by angle
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[int((length - 1) / 2), :] = 1.0
    rot_mat = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
    k_sum = np.sum(kernel)
    if k_sum > 0: kernel /= k_sum

    img_np = np.array(img, dtype=np.uint8)
    out_np = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(out_np)


def apply_dark(img: Image.Image, level: int, rng: np.random.RandomState) -> Image.Image:
    if level <= 0: return img
    # Reduce gamma and illumination factor
    gammas = [1.0, 1.35, 1.75, 2.25, 2.90, 3.70]
    factors = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]
    idx = min(level, 5)
    gamma = gammas[idx]
    factor = factors[idx] * (1.0 + rng.uniform(-0.03, 0.03))

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_dark = (img_np ** gamma) * factor
    img_dark = np.clip(img_dark * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_dark)


def apply_glare(img: Image.Image, level: int, rng: np.random.RandomState) -> Image.Image:
    if level <= 0: return img
    intensities = [0.0, 0.35, 0.50, 0.65, 0.80, 0.95]
    radii_ratio = [0.0, 0.18, 0.25, 0.32, 0.40, 0.48]
    idx = min(level, 5)

    img_np = np.array(img, dtype=np.float32)
    h, w, c = img_np.shape

    # Glare center coordinates
    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))
    sigma = min(h, w) * radii_ratio[idx]

    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx)**2 + (y - cy)**2
    glare_mask = np.exp(-dist_sq / (2.0 * sigma**2))[:, :, np.newaxis]

    glare_val = 255.0 * intensities[idx]
    out_np = img_np + glare_val * glare_mask
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    return Image.fromarray(out_np)


def apply_jpeg(img: Image.Image, level: int, rng: np.random.RandomState) -> Image.Image:
    if level <= 0: return img
    qualities = [95, 50, 32, 18, 10, 4]
    q = qualities[min(level, 5)]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_occlusion(img: Image.Image, level: int, rng: np.random.RandomState) -> Image.Image:
    if level <= 0: return img
    height_ratios = [0.0, 0.10, 0.18, 0.28, 0.38, 0.50]
    hr = height_ratios[min(level, 5)]

    img_np = np.array(img, dtype=np.float32)
    h, w, _ = img_np.shape
    occ_h = int(h * hr)

    # Choose upper eyelid (top) or lower eyelid (bottom) occlusion
    is_top = rng.choice([True, False])
    y_idx = np.arange(h)

    if is_top:
        alpha = np.clip((occ_h - y_idx) / (occ_h * 0.4 + 1e-5), 0.0, 1.0)
    else:
        alpha = np.clip((y_idx - (h - occ_h)) / (occ_h * 0.4 + 1e-5), 0.0, 1.0)

    # Shading effect mimicking eyelid ptosis
    alpha_mask = alpha[:, np.newaxis, np.newaxis]
    out_np = img_np * (1.0 - alpha_mask * 0.95)
    return Image.fromarray(np.clip(out_np, 0, 255).astype(np.uint8))


DEGRADE_DISPATCH = {
    "gaussian_blur": apply_gaussian_blur,
    "defocus": apply_defocus,
    "motion_blur": apply_motion_blur,
    "dark": apply_dark,
    "glare": apply_glare,
    "jpeg": apply_jpeg,
    "occlusion": apply_occlusion,
}


def apply_single_degradation(
    img: Image.Image,
    deg_type: str,
    level: int,
    seed: int,
    sample_id: str,
    view: str = "v1"
) -> Image.Image:
    """Apply a single degradation to an individual image."""
    if level <= 0:
        return img
    fn = DEGRADE_DISPATCH.get(deg_type)
    if fn is None:
        raise ValueError(f"Unsupported degradation type: {deg_type}")
    rng = _get_rng(seed, sample_id, deg_type, level, view)
    return fn(img, level, rng)


def degrade_pair(
    img1: Image.Image,
    img2: Image.Image,
    q1: float,
    q2: float,
    deg_type: str,
    level: int,
    target_view: str,  # "v1", "v2", "both"
    seed: int,
    sample_id: str
) -> Tuple[Image.Image, Image.Image, float, float]:
    """
    Apply controlled degradation to dual-view image pair.
    Concurrently updates images and corresponding oracle quality scores.
    """
    if level <= 0:
        return img1, img2, q1, q2

    # Degradation severity factor s ∈ [0, 1]
    s = float(level) / 5.0

    if target_view in ["v1", "both"]:
        img1 = apply_single_degradation(img1, deg_type, level, seed, sample_id, view="v1")
        q1 = max(0.001, q1 * (1.0 - 0.9 * s))

    if target_view in ["v2", "both"]:
        img2 = apply_single_degradation(img2, deg_type, level, seed, sample_id, view="v2")
        q2 = max(0.001, q2 * (1.0 - 0.9 * s))

    return img1, img2, q1, q2
