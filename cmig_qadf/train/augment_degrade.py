"""
cmig_qadf/train/augment_degrade.py
==================================
Random degradation augmentation for the train_aug training regime (Section 5.2).
Probability p=0.5 per dual-view image pair: randomly selects degradation type, affected view,
and severity level s ∈ [0.1, 1.0].
"""

import random
from PIL import Image
from typing import Tuple
from cmig_qadf.data.degradations import DEGRADATION_TYPES, apply_single_degradation


class OnlineDegradationAugmenter:
    """
    Online stochastic degradation augmentation executed dynamically in training loops.
    """
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        img1: Image.Image,
        img2: Image.Image,
        q1: float,
        q2: float,
        sample_id: str
    ) -> Tuple[Image.Image, Image.Image, float, float]:
        if random.random() > self.prob:
            return img1, img2, q1, q2

        deg_type = random.choice(DEGRADATION_TYPES)
        # Discrete levels 1-5
        level = random.randint(1, 5)
        target = random.choice(["v1", "v2", "both"])
        s = float(level) / 5.0
        seed = random.randint(0, 100000)

        if target in ["v1", "both"]:
            img1 = apply_single_degradation(img1, deg_type, level, seed, sample_id, view="v1")
            q1 = max(0.001, q1 * (1.0 - 0.9 * s))

        if target in ["v2", "both"]:
            img2 = apply_single_degradation(img2, deg_type, level, seed + 1, sample_id, view="v2")
            q2 = max(0.001, q2 * (1.0 - 0.9 * s))

        return img1, img2, q1, q2
