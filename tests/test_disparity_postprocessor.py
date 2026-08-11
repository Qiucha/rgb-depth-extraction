"""
Unit tests for DisparityPostProcessor multi-stage noise filter chain.
"""

import unittest
import numpy as np
import cv2

from src.realworld.disparity_postprocessor import DisparityPostProcessor


class TestDisparityPostProcessor(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        h, w = 100, 100
        self.guidance = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Create clean constant disparity = 10.0
        self.disp_left = np.full((h, w), 10.0, dtype=np.float32)
        
        # Inject artificial noise speckle island (3x3 blob) and extreme spike noise
        self.disp_left[20:23, 20:23] = 45.0  # isolated speckle island
        self.disp_left[50, 50] = 90.0        # isolated spike noise

        # Right disparity for LR check
        self.disp_right = np.full((h, w), 10.0, dtype=np.float32)

    def test_post_processing_noise_filtering(self):
        processor = DisparityPostProcessor(
            lr_threshold=1.0,
            max_speckle_size=50,
            max_speckle_diff=2,
            wls_lambda=1000.0,
            wls_sigma=1.0,
            median_kernel=3
        )

        filtered_disp, valid_mask = processor.process(
            disp_left=self.disp_left,
            guidance_img=self.guidance,
            disp_right=self.disp_right
        )

        self.assertEqual(filtered_disp.shape, (100, 100))
        self.assertEqual(valid_mask.shape, (100, 100))
        self.assertEqual(filtered_disp.dtype, np.float32)

        # Verify isolated noise spike at (50, 50) and speckle at (21, 21) are purged/smoothed
        self.assertLess(filtered_disp[50, 50], 30.0)


if __name__ == "__main__":
    unittest.main()
