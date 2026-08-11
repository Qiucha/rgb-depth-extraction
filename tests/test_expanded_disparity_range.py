"""
Unit tests for Issue #41: Expanded Disparity Range (numDisparities up to 160) & Cost Volume Tuning.
"""

import unittest
import numpy as np
import cv2

from src.realworld.census_sgbm_matcher import CensusSGBMMatcher


class TestExpandedDisparityRange(unittest.TestCase):
    def test_expanded_disparity_range_160(self):
        # Create a synthetic pair with high disparity shift = 140px
        h, w = 240, 480
        img_left = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(img_left, (200, 50), (260, 150), 220, -1)

        img_right = np.zeros((h, w), dtype=np.uint8)
        # Shifted left by 140 pixels in right view (disparity = 140)
        cv2.rectangle(img_right, (60, 50), (120, 150), 220, -1)

        # Matcher with expanded max_disparity = 160
        matcher = CensusSGBMMatcher(window_size=5, min_disparity=0, max_disparity=160)
        self.assertEqual(matcher.num_disparities, 160)

        disparity, valid_mask = matcher.compute_disparity(img_left, img_right)
        self.assertEqual(disparity.shape, (h, w))

        # Inspect cost volume bounds
        diagnostics = matcher.compute_cost_volume_bounds(img_left, img_right)
        self.assertEqual(diagnostics["max_disparity"], 160)
        self.assertEqual(diagnostics["num_disparities"], 160)
        self.assertIn("boundary_saturation_ratio", diagnostics)
        self.assertIn("mean_min_cost", diagnostics)


if __name__ == "__main__":
    unittest.main()
