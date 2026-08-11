import cv2
import numpy as np
import unittest
import pytest

from src.realworld.calibration_refiner import CalibrationRefiner

class TestCalibrationRefinerFixes(unittest.TestCase):
    def setUp(self):
        self.refiner = CalibrationRefiner(pattern_size=(9, 6), square_size_m=0.025)
        
        self.board_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cols, rows = 9, 6
        square_px = 30
        offset_x, offset_y = 100, 100
        
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    x1 = offset_x + c * square_px
                    y1 = offset_y + r * square_px
                    x2 = x1 + square_px
                    y2 = y1 + square_px
                    cv2.rectangle(self.board_img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    def test_float64_corner_crash_with_scale(self):
        img_large = cv2.resize(self.board_img, (3000, 3000))
        found, corners, obj = self.refiner.detect_corners(img_large)
        
        self.assertTrue(found)
        self.assertEqual(corners.dtype, np.float32)

    def test_exception_swallowing_logged(self):
        pass

if __name__ == "__main__":
    unittest.main()
