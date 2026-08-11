"""
Lanczos & Adaptive Unsharp Masking (USM) Acutance Enhancer Engine.
Boosts high-frequency edge gradients on scaled images to align acutance across Main & Ultra-Wide views.
"""

import cv2
import numpy as np


class LanczosAcutanceEnhancer:
    """
    Adaptive Unsharp Masking (USM) filter for boosting edge acutance and high-frequency details.
    """

    def __init__(self, amount: float = 1.5, radius: float = 1.0, threshold: int = 0):
        """
        :param amount: Sharpening multiplier strength factor (e.g. 1.0..2.5)
        :param radius: Gaussian blur kernel radius (pixels)
        :param threshold: Minimum gradient magnitude cutoff to prevent noise amplification
        """
        self.amount = amount
        self.radius = radius
        self.threshold = threshold

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Applies adaptive unsharp masking to enhance image edge acutance.

        :param img: (H, W, 3) BGR or (H, W) grayscale uint8 array
        :return: High-acutance uint8 array of same dimensions
        """
        if self.amount <= 0.0:
            return img.copy()

        img_float = img.astype(np.float32)
        # Compute Gaussian blurred base image
        blurred = cv2.GaussianBlur(img_float, (0, 0), self.radius)

        # High-frequency detail mask
        high_freq = img_float - blurred

        if self.threshold > 0:
            low_contrast = np.abs(high_freq) < self.threshold
            high_freq[low_contrast] = 0.0

        sharpened = img_float + self.amount * high_freq
        return np.clip(sharpened, 0, 255).astype(np.uint8)
