"""
Unit tests for HeterogeneousStereoRectifier fisheye model & principal point alignment.
"""

import unittest
import numpy as np
import cv2

from src.realworld.hetero_rectifier import HeterogeneousStereoRectifier


class TestFisheyeRectifier(unittest.TestCase):

    def setUp(self):
        self.img_main = np.zeros((960, 1280, 3), dtype=np.uint8)
        self.img_uw = np.zeros((960, 1280, 3), dtype=np.uint8)

        # Draw horizontal grid line across both images to verify scanline alignment
        cv2.line(self.img_main, (0, 480), (1280, 480), (255, 255, 255), 2)
        cv2.line(self.img_uw, (0, 480), (1280, 480), (255, 255, 255), 2)

        self.K1 = np.array([[1000.0, 0.0, 640.0],
                            [0.0, 1000.0, 480.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D1 = np.array([-0.1, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)

        self.K2 = np.array([[600.0, 0.0, 640.0],
                            [0.0, 600.0, 485.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D2 = np.array([0.2, -0.05, 0.001, -0.0001], dtype=np.float64)

        self.R = np.eye(3, dtype=np.float64)
        self.T = np.array([[-0.0195], [0.0001], [0.0002]], dtype=np.float64)

    def test_pinhole_rectification_cy_lock(self):
        rectifier = HeterogeneousStereoRectifier(target_size=(1280, 960), is_fisheye=False)
        rect_main, rect_uw, P1, P2, Q = rectifier.rectify_pair(
            self.img_main, self.img_uw,
            self.K1, self.D1, self.K2, self.D2, self.R, self.T
        )

        self.assertEqual(rect_main.shape, (960, 1280, 3))
        self.assertEqual(rect_uw.shape, (960, 1280, 3))
        # Verify strict cy principal point equality
        self.assertAlmostEqual(P1[1, 2], P2[1, 2], places=5)

    def test_fisheye_rectification_cy_lock(self):
        rectifier = HeterogeneousStereoRectifier(target_size=(1280, 960), is_fisheye=True)
        rect_main, rect_uw, P1, P2, Q = rectifier.rectify_pair(
            self.img_main, self.img_uw,
            self.K1, self.D1, self.K2, self.D2, self.R, self.T
        )

        self.assertEqual(rect_main.shape, (960, 1280, 3))
        self.assertEqual(rect_uw.shape, (960, 1280, 3))
        # Verify strict cy principal point equality
        self.assertAlmostEqual(P1[1, 2], P2[1, 2], places=5)


if __name__ == "__main__":
    unittest.main()
