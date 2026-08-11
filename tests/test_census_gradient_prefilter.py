"""
Unit tests for Issue #40: Census & Normalized Gradient Pre-Filtering for Photometric Resilience.
"""

import unittest
import numpy as np
import cv2

from src.realworld.color_normalizer import ColorExposureNormalizer
from src.realworld.census_sgbm_matcher import CensusSGBMMatcher


class TestCensusGradientPrefilter(unittest.TestCase):
    def test_normalized_gradient_image_computation(self):
        # Create a synthetic image with varying brightness gradients
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), 200, -1)

        norm_grad = ColorExposureNormalizer.compute_normalized_gradient_image(img)
        self.assertEqual(norm_grad.shape, (100, 100))
        self.assertEqual(norm_grad.dtype, np.uint8)
        # Edge region should have high gradient response
        self.assertGreater(norm_grad[20, 50], 0)

    def test_photometric_mismatch_resilience(self):
        # Create a stereo pair where right view has severe non-linear gamma/tone curve shift
        img_left = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img_left, (30, 30), (70, 70), 180, -1)

        # Right image shifted by 10 pixels (disparity = 10) and tone mapped (gamma modified)
        img_right = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img_right, (20, 30), (60, 70), 90, -1)  # Photometric mismatch: intensity 90 vs 180

        matcher = CensusSGBMMatcher(window_size=5, min_disparity=0, max_disparity=32)
        disparity, valid_mask = matcher.compute_disparity(img_left, img_right)

        self.assertEqual(disparity.shape, (100, 100))
        self.assertEqual(valid_mask.shape, (100, 100))
        # Central rectangle should show matched disparity ~ 10.0
        rect_disp = disparity[40:60, 40:60]
        valid_rect = rect_disp[rect_disp > 0]
        self.assertGreater(len(valid_rect), 0)
        self.assertAlmostEqual(float(np.median(valid_rect)), 10.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
