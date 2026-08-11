"""
Point Spread Function (PSF) Gradient Mismatch & Lanczos Scale Factor Optimization Engine.
Equalizes high-frequency edge gradient magnitude distributions across heterogeneous camera sensors (Main vs Ultra-Wide)
to eliminate cost volume distortion caused by sensor optical modulation transfer function (MTF) differences.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .acutance_enhancer import LanczosAcutanceEnhancer


class PSFGradientOptimizer:
    """
    Measures and matches high-frequency edge gradient magnitudes between heterogeneous stereo views (Main vs Ultra-Wide).
    """

    def __init__(self, target_ratio: float = 1.0, max_blur_sigma: float = 1.5):
        """
        :param target_ratio: Target gradient magnitude ratio G_main / G_uw (1.0 = equalized).
        :param max_blur_sigma: Upper bound on Gaussian pre-filter smoothing radius.
        """
        self.target_ratio = target_ratio
        self.max_blur_sigma = max_blur_sigma

    @staticmethod
    def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
        """
        Computes 2D Sobel gradient magnitude map for an RGB or grayscale image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def estimate_psf_mismatch(self, img_main: np.ndarray, img_uw: np.ndarray) -> Dict[str, float]:
        """
        Calculates mean gradient magnitude and gradient magnitude ratio gamma = mean(G_main) / mean(G_uw).

        :param img_main: Main camera RGB array.
        :param img_uw: Ultra-Wide camera RGB array.
        :return: Dict containing g_main_mean, g_uw_mean, and gradient_ratio.
        """
        g_main = self.compute_gradient_magnitude(img_main)
        g_uw = self.compute_gradient_magnitude(img_uw)

        mean_main = float(np.mean(g_main))
        mean_uw = float(np.mean(g_uw))
        ratio = mean_main / max(mean_uw, 1e-6)

        return {
            "g_main_mean": round(mean_main, 4),
            "g_uw_mean": round(mean_uw, 4),
            "gradient_ratio": round(ratio, 4)
        }

    def optimize_pair(
        self,
        img_main: np.ndarray,
        img_uw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Adaptively matches high-frequency gradient distributions by smoothing the sharper sensor
        or boosting acutance on the softer sensor.

        :param img_main: Rectified Main image.
        :param img_uw: Rectified Ultra-Wide image.
        :return: (opt_main, opt_uw, telemetry_dict)
        """
        initial_stats = self.estimate_psf_mismatch(img_main, img_uw)
        ratio = initial_stats["gradient_ratio"]

        opt_main = img_main.copy()
        opt_uw = img_uw.copy()
        action_taken = "none"

        if ratio > 1.05:
            # Main camera image has higher sharpness than Ultra-Wide
            # Apply proportional Gaussian smoothing to Main and boost Ultra-Wide acutance
            sigma = min((ratio - 1.0) * 0.5, self.max_blur_sigma)
            opt_main = cv2.GaussianBlur(img_main, (0, 0), sigma)
            action_taken = f"smoothed_main_sigma_{round(sigma, 2)}"

            enhancer = LanczosAcutanceEnhancer(amount=1.2, radius=1.0)
            opt_uw = enhancer.enhance(img_uw)
        elif ratio < 0.95:
            # Ultra-Wide image has higher sharpness than Main
            sigma = min((1.0 / max(ratio, 1e-6) - 1.0) * 0.5, self.max_blur_sigma)
            opt_uw = cv2.GaussianBlur(img_uw, (0, 0), sigma)
            action_taken = f"smoothed_uw_sigma_{round(sigma, 2)}"

            enhancer = LanczosAcutanceEnhancer(amount=1.2, radius=1.0)
            opt_main = enhancer.enhance(img_main)

        final_stats = self.estimate_psf_mismatch(opt_main, opt_uw)

        telemetry = {
            "initial_stats": initial_stats,
            "final_stats": final_stats,
            "action_taken": action_taken
        }

        return opt_main, opt_uw, telemetry
