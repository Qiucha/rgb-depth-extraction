"""
Edge-Preserving Guided Bilateral Depth Filter Module.
Provides joint bilateral filtering on metric depth maps guided by RGB guidance images,
smoothing planar surfaces while locking onto object boundary edges. Supports ON/OFF toggling.
"""

import cv2
import numpy as np
from typing import Tuple


class BilateralDepthFilter:
    """
    Joint/Guided Bilateral Filter for metric depth maps.
    Uses guidance image intensity gradients to prevent depth bleeding across object edges.
    Includes ON/OFF toggle state.
    """

    def __init__(self, window_size: int = 5, sigma_space: float = 3.0, sigma_color: float = 20.0, enabled: bool = True):
        """
        :param window_size: Spatial neighborhood diameter (odd integer, e.g. 5 or 7)
        :param sigma_space: Spatial Gaussian filter standard deviation (pixels)
        :param sigma_color: Color/intensity range Gaussian filter standard deviation (0..255 scale)
        :param enabled: Toggle state (True = apply filter, False = bypass)
        """
        self.window_size = window_size
        self.sigma_space = sigma_space
        self.sigma_color = sigma_color
        self.enabled = enabled

    def filter_depth(self, depth_map: np.ndarray, guidance_img: np.ndarray) -> np.ndarray:
        """
        Applies joint bilateral filter to depth_map using guidance_img.

        :param depth_map: 2D metric depth map float32 array (H, W)
        :param guidance_img: 3-channel BGR/RGB image (H, W, 3) or grayscale (H, W)
        :return: 2D filtered metric depth map float32 array (H, W)
        """
        if not self.enabled:
            return depth_map.copy()

        h, w = depth_map.shape
        rad = self.window_size // 2

        # Convert guidance to single channel float32 luminance (0..255)
        if guidance_img.ndim == 3:
            gray_g = cv2.cvtColor(guidance_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray_g = guidance_img.astype(np.float32)

        depth_f = depth_map.astype(np.float32)

        # Precompute spatial Gaussian weights for (2*rad+1, 2*rad+1) window
        y_grid, x_grid = np.ogrid[-rad:rad+1, -rad:rad+1]
        spatial_kernel = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * self.sigma_space**2)).astype(np.float32)

        accum_depth = np.zeros((h, w), dtype=np.float32)
        accum_weight = np.zeros((h, w), dtype=np.float32)

        # Vectorized 2D loop over spatial window offsets
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                s_weight = spatial_kernel[dy + rad, dx + rad]

                # Shift guidance and depth maps
                shifted_g = np.roll(gray_g, (-dy, -dx), axis=(0, 1))
                shifted_d = np.roll(depth_f, (-dy, -dx), axis=(0, 1))

                # Mask out circular boundary shifts
                valid_mask = np.ones((h, w), dtype=bool)
                if dy < 0:
                    valid_mask[:abs(dy), :] = False
                elif dy > 0:
                    valid_mask[-dy:, :] = False
                if dx < 0:
                    valid_mask[:, :abs(dx)] = False
                elif dx > 0:
                    valid_mask[:, -dx:] = False

                # Range Gaussian weight based on guidance image color intensity difference
                g_diff_sq = (gray_g - shifted_g) ** 2
                range_weight = np.exp(-g_diff_sq / (2.0 * self.sigma_color**2))

                # Only include valid depth pixels (depth > 0)
                d_valid = (shifted_d > 0.0) & valid_mask
                weight = s_weight * range_weight * d_valid.astype(np.float32)

                accum_depth += weight * shifted_d
                accum_weight += weight

        # Normalize accumulated depth
        valid_accum = accum_weight > 1e-5
        filtered_depth = np.zeros((h, w), dtype=np.float32)
        filtered_depth[valid_accum] = accum_depth[valid_accum] / accum_weight[valid_accum]

        # Preserve original zero-depth invalid pixels
        filtered_depth[depth_map <= 0.0] = 0.0

        return filtered_depth
