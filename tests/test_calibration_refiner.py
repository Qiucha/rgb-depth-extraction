"""
Unit tests for CalibrationRefiner module.
"""

import unittest
import numpy as np
import cv2

from src.realworld.calibration_refiner import CalibrationRefiner



class TestCalibrationRefiner(unittest.TestCase):

    def setUp(self):
        self.refiner = CalibrationRefiner(pattern_size=(9, 6), square_size_m=0.025)

        # Generate a synthetic checkerboard image
        h, w = 480, 640
        self.board_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Render clean black-and-white checkerboard squares
        square_px = 30
        start_x, start_y = 100, 80
        for r in range(7):
            for c in range(10):
                if (r + c) % 2 == 0:
                    x1 = start_x + c * square_px
                    y1 = start_y + r * square_px
                    x2 = x1 + square_px
                    y2 = y1 + square_px
                    cv2.rectangle(self.board_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    def test_corner_detection(self):
        found, corners_2d, obj_3d = self.refiner.detect_corners(self.board_img)
        self.assertTrue(found)
        self.assertIsNotNone(corners_2d)
        self.assertEqual(corners_2d.shape[0], 9 * 6)
        self.assertEqual(obj_3d.shape[0], 9 * 6)

    def test_stereo_calibration_fallback_if_no_corners(self):
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        K1 = np.eye(3, dtype=np.float64)
        K2 = np.eye(3, dtype=np.float64)

        result = self.refiner.calibrate_stereo([(blank_img, blank_img)], K1, K2)
        self.assertEqual(result["num_pairs_used"], 0)
        np.testing.assert_array_equal(result["K1"], K1)

    def test_stereo_calibration_refining(self):
        K1 = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        K2 = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)

        result = self.refiner.calibrate_stereo([(self.board_img, self.board_img)], K1, K2)
        self.assertEqual(result["num_pairs_used"], 1)
        self.assertIn("R", result)
        self.assertIn("T", result)

    def test_auto_discover_pattern_size(self):
        # Generate a (5, 4) target checkerboard image
        h, w = 480, 640
        board_5x4 = np.ones((h, w, 3), dtype=np.uint8) * 255
        square_px = 30
        start_x, start_y = 100, 80
        for r in range(5):
            for c in range(6):
                if (r + c) % 2 == 0:
                    cv2.rectangle(board_5x4, (start_x + c * square_px, start_y + r * square_px),
                                  (start_x + (c + 1) * square_px, start_y + (r + 1) * square_px), (0, 0, 0), -1)

        refiner_auto = CalibrationRefiner(pattern_size=(9, 6), square_size_m=0.025, auto_discover=True)
        discovered = refiner_auto.auto_discover_pattern_size([(board_5x4, board_5x4)])
        self.assertIn(discovered, [(5, 4), (4, 5)])

    def test_compute_pair_reprojection_errors(self):
        K1 = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        K2 = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        T = np.array([[-0.0195], [0.0], [0.0]], dtype=np.float64)

        found, corners_2d, obj_3d = self.refiner.detect_corners(self.board_img)
        self.assertTrue(found)

        errors = self.refiner.compute_pair_reprojection_errors(
            [obj_3d], [corners_2d], [corners_2d], K1, D, K2, D, R, T
        )
        self.assertEqual(len(errors), 1)
        self.assertGreaterEqual(errors[0], 0.0)

    def test_compute_pair_reprojection_errors_overflow_safety(self):
        K1 = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        K2 = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        T = np.array([[-0.0195], [0.0], [0.0]], dtype=np.float64)

        found, corners_2d, obj_3d = self.refiner.detect_corners(self.board_img)
        self.assertTrue(found)

        # Corrupt 2D corners with extreme values to test numerical stability & overflow safety
        wild_corners = corners_2d.copy()
        wild_corners[0] = np.array([1e10, 1e10], dtype=np.float32)

        errors = self.refiner.compute_pair_reprojection_errors(
            [obj_3d], [wild_corners], [corners_2d], K1, D, K2, D, R, T
        )
        self.assertEqual(len(errors), 1)
        self.assertTrue(np.isfinite(errors[0]))
        self.assertLessEqual(errors[0], 999.0)


if __name__ == "__main__":
    unittest.main()

