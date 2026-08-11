"""
Unit tests for BilateralDepthFilter module.
"""

import unittest
import numpy as np
import cv2

from src.realworld.bilateral_depth_filter import BilateralDepthFilter


class TestBilateralDepthFilter(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        h, w = 100, 100
        # Guidance image with sharp vertical color boundary at x = 50
        self.guidance = np.zeros((h, w, 3), dtype=np.uint8)
        self.guidance[:, :50] = (255, 0, 0)
        self.guidance[:, 50:] = (0, 0, 255)

        # Depth map matching vertical step boundary (depth = 1.0 on left, 5.0 on right)
        self.depth_map = np.zeros((h, w), dtype=np.float32)
        self.depth_map[:, :50] = 1.0
        self.depth_map[:, 50:] = 5.0

        # Inject surface noise
        noise = np.random.normal(0, 0.1, (h, w)).astype(np.float32)
        self.depth_map += noise

    def test_edge_preserving_bilateral_filter_enabled(self):
        filter_engine = BilateralDepthFilter(window_size=5, sigma_space=3.0, sigma_color=20.0, enabled=True)
        filtered = filter_engine.filter_depth(self.depth_map, self.guidance)

        self.assertEqual(filtered.shape, (100, 100))
        self.assertEqual(filtered.dtype, np.float32)

        # Check edge boundary preservation: left side mean should be near 1.0, right side near 5.0
        left_mean = np.mean(filtered[10:-10, 10:45])
        right_mean = np.mean(filtered[10:-10, 55:90])

        self.assertAlmostEqual(left_mean, 1.0, delta=0.2)
        self.assertAlmostEqual(right_mean, 5.0, delta=0.2)

    def test_bilateral_filter_disabled_toggle(self):
        filter_engine = BilateralDepthFilter(enabled=False)
        bypassed = filter_engine.filter_depth(self.depth_map, self.guidance)

        np.testing.assert_array_equal(bypassed, self.depth_map)


if __name__ == "__main__":
    unittest.main()
