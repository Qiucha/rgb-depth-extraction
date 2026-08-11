"""
Unit tests for CensusSGBMMatcher backend.
"""

import unittest
import numpy as np
import cv2

from src.realworld.census_sgbm_matcher import CensusSGBMMatcher
from src.realworld.deep_matcher import get_stereo_matcher


class TestCensusSGBMMatcher(unittest.TestCase):

    def setUp(self):
        # Create synthetic left/right pair with shift = 8 pixels
        np.random.seed(42)
        h, w = 120, 160
        self.left = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Shift right image by 8 pixels to the left (disparity = 8)
        self.right = np.zeros_like(self.left)
        self.right[:, :-8] = self.left[:, 8:]

    def test_census_sgbm_disparity(self):
        matcher = CensusSGBMMatcher(window_size=5, min_disparity=0, max_disparity=32)
        disp, mask = matcher.compute_disparity(self.left, self.right)

        self.assertEqual(disp.shape, (120, 160))
        self.assertEqual(mask.shape, (120, 160))
        self.assertEqual(disp.dtype, np.float32)

        # Mean disparity in valid interior should be near 8.0
        interior_disp = disp[10:-10, 10:-10]
        valid_interior = interior_disp[interior_disp > 0]
        if len(valid_interior) > 0:
            self.assertAlmostEqual(np.median(valid_interior), 8.0, delta=2.0)

    def test_factory_registration(self):
        matcher = get_stereo_matcher("census_sgbm", window_size=5, max_disparity=32)
        self.assertIsInstance(matcher, CensusSGBMMatcher)


if __name__ == "__main__":
    unittest.main()
