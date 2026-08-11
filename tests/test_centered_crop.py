"""
Unit tests for Centered Ultra-Wide Lens Crop & Physical Baseline Alignment.
"""

import unittest
import numpy as np

from src.realworld.hetero_rectifier import HeterogeneousStereoRectifier


class TestCenteredCropAndBaseline(unittest.TestCase):

    def setUp(self):
        self.rectifier = HeterogeneousStereoRectifier(target_size=(1280, 960), is_fisheye=False)

        # Standard iPhone 1x Wide intrinsics (1280x960)
        self.K1 = np.array([
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 480.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.D1 = np.zeros(5, dtype=np.float64)

        # Ultra-Wide 0.5x intrinsics (1280x960)
        self.K2 = np.array([
            [500.0, 0.0, 640.0],
            [0.0, 500.0, 480.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.D2 = np.zeros(5, dtype=np.float64)

        # Physical 19.5mm baseline vector
        self.R = np.eye(3, dtype=np.float64)
        self.T = np.array([[-0.0195], [0.0], [0.0]], dtype=np.float64)

        self.img1 = np.zeros((960, 1280, 3), dtype=np.uint8)
        self.img2 = np.zeros((960, 1280, 3), dtype=np.uint8)

    def test_principal_point_optical_center_alignment(self):
        rect_main, rect_uw, P1, P2, Q = self.rectifier.rectify_pair(
            self.img1, self.img2, self.K1, self.D1, self.K2, self.D2, self.R, self.T
        )

        # P1[0, 2] and P2[0, 2] cx must be identical
        self.assertAlmostEqual(P1[0, 2], P2[0, 2], places=5)
        # P1[1, 2] and P2[1, 2] cy must be identical
        self.assertAlmostEqual(P1[1, 2], P2[1, 2], places=5)

    def test_physical_baseline_locking(self):
        rect_main, rect_uw, P1, P2, Q = self.rectifier.rectify_pair(
            self.img1, self.img2, self.K1, self.D1, self.K2, self.D2, self.R, self.T
        )

        f_rect = P1[0, 0]
        baseline_calc = abs(P2[0, 3] / f_rect)
        self.assertAlmostEqual(baseline_calc, 0.0195, places=5)


if __name__ == "__main__":
    unittest.main()
