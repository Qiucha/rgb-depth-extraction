"""
Multi-Stage Disparity Post-Processing & Noise Reduction Filter Chain.
Applies Left-Right consistency validation, OpenCV speckle filtering, guided WLS smoothing,
guided Bilateral Depth filtering, and confidence median filtering with explicit ON/OFF toggles.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from src.stereo_depth.wls_filter import WLSDisparityFilter
from .bilateral_depth_filter import BilateralDepthFilter


class DisparityPostProcessor:
    """
    Multi-pass post-processor for disparity and metric depth maps.
    Supports individual ON/OFF toggling for all filter passes.
    """

    def __init__(
        self,
        lr_threshold: float = 1.0,
        max_speckle_size: int = 100,
        max_speckle_diff: int = 2,
        wls_lambda: float = 8000.0,
        wls_sigma: float = 1.5,
        median_kernel: int = 3,
        enable_lr_check: bool = True,
        enable_speckle: bool = True,
        enable_wls: bool = True,
        enable_median: bool = True,
        enable_bilateral: bool = True,
        bilateral_window: int = 5,
        bilateral_sigma_space: float = 3.0,
        bilateral_sigma_color: float = 20.0
    ):
        """
        :param lr_threshold: Maximum allowed Left-Right cross-check disparity difference (pixels)
        :param max_speckle_size: Maximum blob size to consider as noise speckle (pixels)
        :param max_speckle_diff: Maximum disparity difference within speckle blob
        :param wls_lambda: Smoothness weight parameter for WLS filter
        :param wls_sigma: Color intensity sensitivity parameter for WLS filter
        :param median_kernel: Kernel size for residual median filter (e.g. 3 or 5)
        :param enable_lr_check: Toggle Left-Right consistency check
        :param enable_speckle: Toggle OpenCV speckle removal filter
        :param enable_wls: Toggle WLS edge-preserving smoothing filter
        :param enable_median: Toggle 3x3 median filter
        :param enable_bilateral: Toggle guided bilateral depth filter
        """
        self.lr_threshold = lr_threshold
        self.max_speckle_size = max_speckle_size
        self.max_speckle_diff = max_speckle_diff
        self.wls_filter = WLSDisparityFilter(lambda_val=wls_lambda, sigma_val=wls_sigma)
        self.median_kernel = median_kernel

        self.enable_lr_check = enable_lr_check
        self.enable_speckle = enable_speckle
        self.enable_wls = enable_wls
        self.enable_median = enable_median

        self.bilateral_filter = BilateralDepthFilter(
            window_size=bilateral_window,
            sigma_space=bilateral_sigma_space,
            sigma_color=bilateral_sigma_color,
            enabled=enable_bilateral
        )

    def process(
        self,
        disp_left: np.ndarray,
        guidance_img: np.ndarray,
        disp_right: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes multi-stage filtering pipeline on raw left disparity map.

        :param disp_left: Raw left disparity map (H, W) float32
        :param guidance_img: Rectified left RGB image (H, W, 3) or grayscale (H, W)
        :param disp_right: Optional raw right disparity map (H, W) float32 for LR check
        :param valid_mask: Optional input boolean validity mask (H, W)
        :return: Tuple of (filtered_disparity [float32], final_valid_mask [bool])
        """
        h, w = disp_left.shape
        filtered_disp = disp_left.copy()
        out_mask = valid_mask.copy() if valid_mask is not None else (disp_left > 0)

        # -------------------------------------------------------------
        # Pass 1: Left-Right Cross Check Invalidation
        # -------------------------------------------------------------
        if self.enable_lr_check and disp_right is not None:
            y_grid, x_grid = np.indices((h, w))
            xr_grid = np.round(x_grid - disp_left).astype(np.int32)

            in_bounds = (xr_grid >= 0) & (xr_grid < w)
            y_v = y_grid[in_bounds]
            x_v = x_grid[in_bounds]
            xr_v = xr_grid[in_bounds]

            d_l = disp_left[y_v, x_v]
            d_r = disp_right[y_v, xr_v]

            lr_valid = np.abs(d_l - d_r) <= self.lr_threshold
            
            # Invalidate pixels failing LR check
            out_mask[y_v[~lr_valid], x_v[~lr_valid]] = False
            filtered_disp[~out_mask] = 0.0

        # -------------------------------------------------------------
        # Pass 2: Speckle Invalidation Filtering (cv2.filterSpeckles)
        # -------------------------------------------------------------
        if self.enable_speckle and self.max_speckle_size > 0:
            disp_int16 = np.round(filtered_disp * 16.0).astype(np.int16)
            disp_int16[~out_mask] = 0

            cv2.filterSpeckles(
                img=disp_int16,
                newVal=0,
                maxSpeckleSize=self.max_speckle_size,
                maxDiff=self.max_speckle_diff * 16
            )

            speckle_valid = disp_int16 > 0
            out_mask &= speckle_valid
            filtered_disp = disp_int16.astype(np.float32) / 16.0

        # -------------------------------------------------------------
        # Pass 3: Guided Edge-Preserving WLS Smoothing
        # -------------------------------------------------------------
        if self.enable_wls:
            gray_guidance = cv2.cvtColor(guidance_img, cv2.COLOR_BGR2GRAY) if guidance_img.ndim == 3 else guidance_img
            wls_filtered = self.wls_filter.filter(filtered_disp, gray_guidance)
        else:
            wls_filtered = filtered_disp.copy()

        # -------------------------------------------------------------
        # Pass 4: Guided Bilateral Depth Filtering
        # -------------------------------------------------------------
        if self.bilateral_filter.enabled:
            wls_filtered = self.bilateral_filter.filter_depth(wls_filtered, guidance_img)

        # -------------------------------------------------------------
        # Pass 5: Confidence Masking & 3x3 Median Filtering
        # -------------------------------------------------------------
        if self.enable_median and self.median_kernel > 1:
            median_disp = cv2.medianBlur(wls_filtered.astype(np.float32), self.median_kernel)
            final_disp = np.where(out_mask, median_disp, 0.0)
        else:
            final_disp = np.where(out_mask, wls_filtered, 0.0)

        return final_disp.astype(np.float32), out_mask
