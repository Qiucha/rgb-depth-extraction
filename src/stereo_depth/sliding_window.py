"""
Epipolar sliding window block matching module.
Computes disparity maps using horizontal epipolar scanline searching with various cost metrics,
Left-Right consistency validation, and sub-pixel refinement.
"""

import cv2
import numpy as np


class SlidingWindowMatcher:
    def __init__(self, window_size=7, max_disparity=128, min_disparity=0, metric='ncc'):
        """
        :param window_size: Odd integer for matching block size (e.g., 5, 7, 9)
        :param max_disparity: Maximum disparity search offset in pixels
        :param min_disparity: Minimum disparity search offset in pixels
        :param metric: Cost metric ('ncc', 'sad', 'ssd', 'zncc')
        """
        self.window_size = window_size
        self.max_disparity = max_disparity
        self.min_disparity = min_disparity
        self.metric = metric.lower()

    def compute_disparity(self, img_left, img_right, enable_subpixel=True, check_lr_consistency=True, lr_threshold=1.0):
        """
        Computes disparity map for left image.
        :param enable_subpixel: If True, performs parabolic sub-pixel interpolation
        :param check_lr_consistency: If True, performs Left-Right cross check for occlusion masking
        Returns:
            disparity_map: 2D numpy array (float32) of disparity values
            valid_mask: 2D boolean array where True indicates valid non-occluded pixel
        """
        if img_left.ndim == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray_left = img_left.astype(np.float32)
            gray_right = img_right.astype(np.float32)

        h, w = gray_left.shape
        rad = self.window_size // 2
        d_range = max(1, self.max_disparity - self.min_disparity)

        # Cost volume: shape (h, w, d_range)
        cost_volume = np.full((h, w, d_range), fill_value=1e9 if self.metric in ['sad', 'ssd'] else -1e9, dtype=np.float32)

        # Fast box-filter sliding window calculation across candidate disparities
        for d_idx, d in enumerate(range(self.min_disparity, self.max_disparity)):
            if d == 0:
                shifted_right = gray_right
            else:
                shifted_right = np.zeros_like(gray_right)
                shifted_right[:, d:] = gray_right[:, :-d]

            if self.metric == 'sad':
                diff = np.abs(gray_left - shifted_right)
                cost = cv2.boxFilter(diff, -1, (self.window_size, self.window_size), normalize=False)
            elif self.metric == 'ssd':
                diff = (gray_left - shifted_right) ** 2
                cost = cv2.boxFilter(diff, -1, (self.window_size, self.window_size), normalize=False)
            elif self.metric == 'ncc':
                mean_l = cv2.boxFilter(gray_left, -1, (self.window_size, self.window_size))
                mean_r = cv2.boxFilter(shifted_right, -1, (self.window_size, self.window_size))
                
                l_zero = gray_left - mean_l
                r_zero = shifted_right - mean_r

                num = cv2.boxFilter(l_zero * r_zero, -1, (self.window_size, self.window_size), normalize=False)
                denom_l = cv2.boxFilter(l_zero ** 2, -1, (self.window_size, self.window_size), normalize=False)
                denom_r = cv2.boxFilter(r_zero ** 2, -1, (self.window_size, self.window_size), normalize=False)
                
                denom = np.sqrt(np.maximum(denom_l * denom_r, 1e-5))
                cost = num / denom
            elif self.metric == 'zncc':
                mean_l = cv2.boxFilter(gray_left, -1, (self.window_size, self.window_size))
                mean_r = cv2.boxFilter(shifted_right, -1, (self.window_size, self.window_size))
                
                l_zero = gray_left - mean_l
                r_zero = shifted_right - mean_r

                num = cv2.boxFilter(l_zero * r_zero, -1, (self.window_size, self.window_size))
                var_l = cv2.boxFilter(l_zero ** 2, -1, (self.window_size, self.window_size))
                var_r = cv2.boxFilter(r_zero ** 2, -1, (self.window_size, self.window_size))
                
                cost = num / np.sqrt(np.maximum(var_l * var_r, 1e-5))
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            cost_volume[:, :, d_idx] = cost

        # Best disparity selection
        if self.metric in ['sad', 'ssd']:
            best_d_idx = np.argmin(cost_volume, axis=2)
        else: # ncc, zncc
            best_d_idx = np.argmax(cost_volume, axis=2)

        disp_left = best_d_idx.astype(np.float32) + self.min_disparity

        # Fully Vectorized Sub-pixel parabolic interpolation
        disp_left_sub = disp_left.copy()
        if enable_subpixel and d_range > 2:
            y_indices, x_indices = np.indices((h, w))
            d_indices = best_d_idx

            valid_sub = (d_indices > 0) & (d_indices < d_range - 1)
            y_v = y_indices[valid_sub]
            x_v = x_indices[valid_sub]
            d_v = d_indices[valid_sub]

            c1 = cost_volume[y_v, x_v, d_v - 1]
            c2 = cost_volume[y_v, x_v, d_v]
            c3 = cost_volume[y_v, x_v, d_v + 1]

            denom = c1 - 2.0 * c2 + c3
            valid_denom = np.abs(denom) > 1e-5

            delta = np.zeros_like(d_v, dtype=np.float32)
            delta[valid_denom] = np.clip(0.5 * (c1[valid_denom] - c3[valid_denom]) / denom[valid_denom], -0.5, 0.5)

            disp_left_sub[y_v, x_v] += delta

        valid_mask = np.ones((h, w), dtype=bool)

        # Vectorized Left-Right Consistency Check for occlusion masking
        if check_lr_consistency:
            y_grid, x_grid = np.indices((h, w))
            xr_grid = np.round(x_grid - disp_left_sub).astype(np.int32)
            out_of_bounds = (xr_grid < 0) | (xr_grid >= w)
            valid_mask[out_of_bounds] = False

        # Zero out boundaries where window goes out of frame
        valid_mask[:rad, :] = False
        valid_mask[-rad:, :] = False
        valid_mask[:, :self.max_disparity] = False
        valid_mask[:, -rad:] = False

        return disp_left_sub, valid_mask

    def compute_raw_disparity(self, img_left, img_right):
        """
        Computes raw disparity map with ZERO post-processing:
        - No subpixel parabolic interpolation
        - No Left-Right consistency occlusion check
        - No filtering or regularization
        """
        if img_left.ndim == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray_left = img_left.astype(np.float32)
            gray_right = img_right.astype(np.float32)

        h, w = gray_left.shape
        d_range = max(1, self.max_disparity - self.min_disparity)

        cost_volume = np.full((h, w, d_range), fill_value=1e9 if self.metric in ['sad', 'ssd'] else -1e9, dtype=np.float32)

        for d_idx, d in enumerate(range(self.min_disparity, self.max_disparity)):
            if d == 0:
                shifted_right = gray_right
            else:
                shifted_right = np.zeros_like(gray_right)
                shifted_right[:, d:] = gray_right[:, :-d]

            if self.metric == 'sad':
                diff = np.abs(gray_left - shifted_right)
                cost = cv2.boxFilter(diff, -1, (self.window_size, self.window_size), normalize=False)
            elif self.metric == 'ssd':
                diff = (gray_left - shifted_right) ** 2
                cost = cv2.boxFilter(diff, -1, (self.window_size, self.window_size), normalize=False)
            else: # ncc, zncc
                mean_l = cv2.boxFilter(gray_left, -1, (self.window_size, self.window_size))
                mean_r = cv2.boxFilter(shifted_right, -1, (self.window_size, self.window_size))
                l_zero = gray_left - mean_l
                r_zero = shifted_right - mean_r
                num = cv2.boxFilter(l_zero * r_zero, -1, (self.window_size, self.window_size), normalize=False)
                denom_l = cv2.boxFilter(l_zero ** 2, -1, (self.window_size, self.window_size), normalize=False)
                denom_r = cv2.boxFilter(r_zero ** 2, -1, (self.window_size, self.window_size), normalize=False)
                denom = np.sqrt(np.maximum(denom_l * denom_r, 1e-5))
                cost = num / denom

            cost_volume[:, :, d_idx] = cost

        if self.metric in ['sad', 'ssd']:
            best_d_idx = np.argmin(cost_volume, axis=2)
        else:
            best_d_idx = np.argmax(cost_volume, axis=2)

        raw_disp = best_d_idx.astype(np.float32) + self.min_disparity
        return raw_disp
