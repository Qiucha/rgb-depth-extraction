"""
Unit tests for LanczosAcutanceEnhancer module.
"""

import unittest
import numpy as np
import cv2

from src.realworld.acutance_enhancer import LanczosAcutanceEnhancer


class TestAcutanceEnhancer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        h, w = 100, 100
        # Create image with a soft edge gradient in the center
        self.img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(self.img, (50, 50), 30, (200, 200, 200), -1)
        self.img = cv2.GaussianBlur(self.img, (5, 5), 1.0)

    def test_acutance_enhancement_boosts_gradients(self):
        enhancer = LanczosAcutanceEnhancer(amount=1.5, radius=1.0, threshold=0)
        enhanced = enhancer.enhance(self.img)

        self.assertEqual(enhanced.shape, self.img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

        # Compute Sobel gradient magnitudes before and after
        grad_orig = cv2.Sobel(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        grad_enh = cv2.Sobel(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)

        # Enhanced image should have higher max edge gradient magnitude
        self.assertGreater(np.max(np.abs(grad_enh)), np.max(np.abs(grad_orig)))

    def test_zero_amount_bypasses(self):
        enhancer = LanczosAcutanceEnhancer(amount=0.0)
        bypassed = enhancer.enhance(self.img)
        np.testing.assert_array_equal(bypassed, self.img)


if __name__ == "__main__":
    unittest.main()
