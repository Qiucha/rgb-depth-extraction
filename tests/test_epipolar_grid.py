"""
Unit tests for Issue #39: 20px Epipolar Grid Overlay Verification Engine & Digest Visualizer.
"""

import os
import tempfile
import unittest
import numpy as np
import cv2

from src.realworld.digest_builder import generate_realworld_digest


class TestEpipolarGrid(unittest.TestCase):
    def test_epipolar_grid_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "digest_epipolar_test")
            h, w = 480, 640
            rect_main = np.zeros((h, w, 3), dtype=np.uint8)
            rect_uw = np.ones((h, w, 3), dtype=np.uint8) * 128
            disparity = np.ones((h, w), dtype=np.float32) * 10.0
            depth = np.ones((h, w), dtype=np.float32) * 1.5

            scene_name = "Epipolar Grid Test"
            generate_realworld_digest(
                rect_main=rect_main,
                rect_uw=rect_uw,
                disparity_map=disparity,
                depth_map_m=depth,
                focal_length_px=500.0,
                baseline_m=0.02,
                output_dir=output_dir,
                scene_name=scene_name
            )

            asset_dir = os.path.join(output_dir, "assets", scene_name.replace(" ", "_"))
            grid_img_path = os.path.join(asset_dir, "epipolar_grid.jpg")
            self.assertTrue(os.path.exists(grid_img_path), "epipolar_grid.jpg was not generated in assets")

            grid_img = cv2.imread(grid_img_path)
            self.assertIsNotNone(grid_img)
            self.assertEqual(grid_img.shape[0], h)
            self.assertEqual(grid_img.shape[1], w * 2)

            # Check horizontal line drawn at y=0
            self.assertFalse(np.array_equal(grid_img[0, 100], [0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
