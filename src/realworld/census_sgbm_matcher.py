"""
Census-SGBM Stereo Matching Backend Module.
Combines 5x5 Census Transform bitmask representations with Semi-Global Block Matching (SGBM).
"""

import cv2
import numpy as np
from typing import Tuple

from .deep_matcher import BaseStereoMatcher, ensure_grayscale
from .color_normalizer import ColorExposureNormalizer


class CensusSGBMMatcher(BaseStereoMatcher):
    """
    Stereo matcher executing 5x5 Census Transform preprocessing and
    Semi-Global Block Matching (SGBM) 8-path dynamic programming.
    """

    def __init__(self, window_size: int = 5, min_disparity: int = 0, max_disparity: int = 128, uniqueness_ratio: int = 10):
        """
        :param window_size: Matching block size (odd integer, e.g. 5 or 7)
        :param min_disparity: Minimum disparity search bound
        :param max_disparity: Maximum disparity search bound
        :param uniqueness_ratio: Percentage margin by which best cost must beat second best (e.g. 10)
        """
        self.window_size = window_size
        self.min_disparity = min_disparity
        num_disp = max(16, ((max_disparity - min_disparity) + 15) // 16 * 16)
        self.num_disparities = num_disp
        self.uniqueness_ratio = uniqueness_ratio

        # SGBM smoothness penalties
        p1 = 8 * 1 * (window_size ** 2)
        p2 = 32 * 1 * (window_size ** 2)

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.window_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Census-SGBM disparity map and validity mask.

        :param img_left: Left rectified image (H, W) or (H, W, 3)
        :param img_right: Right rectified image (H, W) or (H, W, 3)
        :return: (disparity_map [float32], valid_mask [bool])
        """
        gray_l = ensure_grayscale(img_left)
        gray_r = ensure_grayscale(img_right)

        # 1. Luminance & CDF Normalization
        normalizer = ColorExposureNormalizer(method="cdf")
        norm_r = normalizer.match_luminance(gray_r, gray_l)

        # 2. Compute 5x5 Census Transform bitmasks & normalized gradient pre-filtered images
        census_l = ColorExposureNormalizer.compute_census_transform_5x5(gray_l)
        census_r = ColorExposureNormalizer.compute_census_transform_5x5(norm_r)

        grad_l = ColorExposureNormalizer.compute_normalized_gradient_image(gray_l)
        grad_r = ColorExposureNormalizer.compute_normalized_gradient_image(norm_r)

        # 3. Compute SGBM Disparity map on normalized gradient pre-filtered inputs
        # OpenCV StereoSGBM returns 16-bit fixed point disparities (divide by 16.0)
        disp_16bit = self.sgbm.compute(grad_l, grad_r)
        disparity = disp_16bit.astype(np.float32) / 16.0

        # Build validity mask
        valid_mask = (disparity > self.min_disparity) & (disparity < self.min_disparity + self.num_disparities)

        return disparity, valid_mask

    def compute_cost_volume_bounds(self, img_left: np.ndarray, img_right: np.ndarray) -> dict:
        """
        Inspects Census cost volume bounds across candidate disparities [min_disparity..max_disparity].
        Returns diagnostic statistics and boundary saturation ratio.
        """
        gray_l = ensure_grayscale(img_left)
        gray_r = ensure_grayscale(img_right)

        normalizer = ColorExposureNormalizer(method="cdf")
        norm_r = normalizer.match_luminance(gray_r, gray_l)

        census_l = ColorExposureNormalizer.compute_census_transform_5x5(gray_l)
        census_r = ColorExposureNormalizer.compute_census_transform_5x5(norm_r)

        cost_vol = ColorExposureNormalizer.compute_census_cost_volume(
            census_l, census_r,
            max_disparity=self.min_disparity + self.num_disparities,
            min_disparity=self.min_disparity
        )

        min_cost_indices = np.argmin(cost_vol, axis=2)
        max_idx = cost_vol.shape[2] - 1
        boundary_saturation_ratio = float(np.mean(min_cost_indices == max_idx))

        return {
            "num_disparities": self.num_disparities,
            "min_disparity": self.min_disparity,
            "max_disparity": self.min_disparity + self.num_disparities,
            "boundary_saturation_ratio": boundary_saturation_ratio,
            "mean_min_cost": float(np.mean(np.min(cost_vol, axis=2))),
            "cost_volume_shape": cost_vol.shape
        }

