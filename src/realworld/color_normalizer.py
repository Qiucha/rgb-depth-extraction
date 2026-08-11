"""
Color & Exposure Normalization Preprocessing Engine for Stereo Matching.
Provides histogram/luminance matching (CDF & mean-variance) and Census Transform computation.
"""

import cv2
import numpy as np
from typing import Tuple, Union


class ColorExposureNormalizer:
    """
    Handles luminance/contrast normalization and 5x5 Census Transform bitmask extraction
    for heterogeneous stereo pairs with exposure/color mismatches.
    """

    def __init__(self, method: str = "cdf"):
        """
        :param method: Normalization method ('cdf', 'mean_std', or 'none').
        """
        self.method = method.lower()

    def match_luminance(self, target_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
        """
        Adjusts target_img luminance/color to match ref_img reference distribution.

        :param target_img: Image to adjust (H, W, C) or (H, W)
        :param ref_img: Reference image (H, W, C) or (H, W)
        :return: Normalized target image matching reference brightness distribution.
        """
        if self.method == "none":
            return target_img.copy()

        is_color = (target_img.ndim == 3 and target_img.shape[2] == 3)
        if is_color:
            tgt_ycrcb = cv2.cvtColor(target_img, cv2.COLOR_BGR2YCrCb)
            ref_ycrcb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2YCrCb)

            tgt_y = tgt_ycrcb[:, :, 0]
            ref_y = ref_ycrcb[:, :, 0]

            matched_y = self._match_channel(tgt_y, ref_y)
            tgt_ycrcb[:, :, 0] = matched_y
            return cv2.cvtColor(tgt_ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return self._match_channel(target_img, ref_img)

    def _match_channel(self, tgt: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Matches 2D single-channel intensity distribution from tgt to ref.
        """
        if self.method == "mean_std":
            tgt_f = tgt.astype(np.float32)
            ref_f = ref.astype(np.float32)

            mu_t, std_t = np.mean(tgt_f), np.std(tgt_f)
            mu_r, std_r = np.mean(ref_f), np.std(ref_f)

            if std_t < 1e-5:
                return tgt.copy()

            matched = (tgt_f - mu_t) * (std_r / std_t) + mu_r
            return np.clip(matched, 0, 255).astype(np.uint8)

        else:  # cdf histogram matching
            tgt_counts, _ = np.histogram(tgt.flatten(), bins=256, range=(0, 256))
            ref_counts, _ = np.histogram(ref.flatten(), bins=256, range=(0, 256))

            tgt_cdf = np.cumsum(tgt_counts).astype(np.float32)
            tgt_cdf /= (tgt_cdf[-1] + 1e-5)

            ref_cdf = np.cumsum(ref_counts).astype(np.float32)
            ref_cdf /= (ref_cdf[-1] + 1e-5)

            # Map target intensity levels (0..255) to reference intensity levels with matching CDF
            lookup_table = np.zeros(256, dtype=np.uint8)
            ref_idx = 0
            for i in range(256):
                while ref_idx < 255 and ref_cdf[ref_idx] < tgt_cdf[i]:
                    ref_idx += 1
                lookup_table[i] = ref_idx

            return cv2.LUT(tgt, lookup_table)

    @staticmethod
    def compute_census_transform_5x5(img: np.ndarray) -> np.ndarray:
        """
        Computes 5x5 Census Transform bitmask for each pixel.
        Each 5x5 window has 24 surrounding neighbors comparing intensity to center pixel.
        Bit is set to 1 if neighbor < center, else 0. Packed into 32-bit integer array.

        :param img: 2D uint8 or float32 grayscale image (H, W).
        :return: 2D uint32 numpy array (H, W) storing 24-bit Census codes.
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray = gray.astype(np.uint8)
        h, w = gray.shape
        census = np.zeros((h, w), dtype=np.uint32)

        # 5x5 neighborhood offsets (excluding center (0,0))
        offsets = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2),  (0, -1),           (0, 1),  (0, 2),
            (1, -2),  (1, -1),  (1, 0),  (1, 1),  (1, 2),
            (2, -2),  (2, -1),  (2, 0),  (2, 1),  (2, 2),
        ]

        # Compute bitmask using vectorized array rolling/padding
        center = gray.astype(np.int16)
        bit_idx = 0

        for dy, dx in offsets:
            # Shift neighbor image
            shifted = np.roll(gray, (-dy, -dx), axis=(0, 1)).astype(np.int16)
            
            # Mask out invalid boundary roll values
            valid_mask = np.ones((h, w), dtype=bool)
            if dy < 0:
                valid_mask[:abs(dy), :] = False
            elif dy > 0:
                valid_mask[-dy:, :] = False
            if dx < 0:
                valid_mask[:, :abs(dx)] = False
            elif dx > 0:
                valid_mask[:, -dx:] = False

            bit_comparison = (shifted < center) & valid_mask
            census |= (bit_comparison.astype(np.uint32) << bit_idx)
            bit_idx += 1

        return census

    @staticmethod
    def compute_census_cost_volume(census_left: np.ndarray, census_right: np.ndarray, max_disparity: int, min_disparity: int = 0) -> np.ndarray:
        """
        Computes Census Hamming distance cost volume across candidates disparities.

        :param census_left: (H, W) uint32 Census codes for left frame
        :param census_right: (H, W) uint32 Census codes for right frame
        :param max_disparity: Maximum search offset
        :param min_disparity: Minimum search offset
        :return: 3D float32 numpy array (H, W, D) storing Hamming distance costs.
        """
        h, w = census_left.shape
        d_range = max(1, max_disparity - min_disparity)
        cost_volume = np.zeros((h, w, d_range), dtype=np.float32)

        # Lookup table for popcount (number of set bits in 24 bits)
        # Using uint32 popcount via vector bit ops
        for d_idx, d in enumerate(range(min_disparity, max_disparity)):
            if d == 0:
                shifted_r = census_right
            else:
                shifted_r = np.zeros_like(census_right)
                shifted_r[:, d:] = census_right[:, :-d]

            xor_mask = np.bitwise_xor(census_left, shifted_r)
            
            # Count set bits in uint32 using bit manipulation
            # 24-bit popcount: x - ((x >> 1) & 0x55555555) ...
            v = xor_mask
            v = v - ((v >> np.uint32(1)) & np.uint32(0x55555555))
            v = (v & np.uint32(0x33333333)) + ((v >> np.uint32(2)) & np.uint32(0x33333333))
            popcount = (((v + (v >> np.uint32(4))) & np.uint32(0x0F0F0F0F)) * np.uint32(0x01010101)) >> np.uint32(24)

            cost_volume[:, :, d_idx] = popcount.astype(np.float32)

        return cost_volume

    @staticmethod
    def compute_normalized_gradient_image(img: np.ndarray) -> np.ndarray:
        """
        Computes normalized Sobel gradient image (0..255 uint8) invariant to global/local ISP tone mapping.
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray_f = gray.astype(np.float32)
        sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Local variance normalization using 5x5 box filter
        mean = cv2.boxFilter(gray_f, -1, (5, 5))
        mean_sq = cv2.boxFilter(gray_f ** 2, -1, (5, 5))
        var = np.maximum(mean_sq - mean ** 2, 1e-3)
        std = np.sqrt(var)

        norm_grad = magnitude / (std + 10.0)
        norm_grad_uint8 = np.clip(norm_grad * 10.0, 0, 255).astype(np.uint8)
        return norm_grad_uint8

