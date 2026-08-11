"""
Unit test suite for PSFGradientOptimizer (Issue #43).
Verifies high-frequency gradient magnitude computation, PSF mismatch estimation,
and adaptive gradient equalization between Main and Ultra-Wide views.
"""

import unittest
import numpy as np
import cv2
from src.realworld.psf_gradient_optimizer import PSFGradientOptimizer


class TestPSFGradientOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = PSFGradientOptimizer(target_ratio=1.0)
        # Create synthetic test patterns with high-frequency edge texture
        np.random.seed(42)
        self.img_sharp = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

        # Create a blurred version simulating a softer sensor PSF
        self.img_soft = cv2.GaussianBlur(self.img_sharp, (9, 9), 2.5)

    def test_compute_gradient_magnitude(self):
        g = self.optimizer.compute_gradient_magnitude(self.img_sharp)
        self.assertEqual(g.shape, (100, 100))
        self.assertGreater(np.max(g), 0.0)

    def test_estimate_psf_mismatch(self):
        stats = self.optimizer.estimate_psf_mismatch(self.img_sharp, self.img_soft)
        self.assertIn("g_main_mean", stats)
        self.assertIn("g_uw_mean", stats)
        self.assertIn("gradient_ratio", stats)
        # Sharp image should have higher mean gradient magnitude than soft image
        self.assertGreater(stats["gradient_ratio"], 1.0)

    def test_optimize_pair_reduces_gradient_mismatch(self):
        opt_main, opt_uw, telemetry = self.optimizer.optimize_pair(self.img_sharp, self.img_soft)
        self.assertEqual(opt_main.shape, self.img_sharp.shape)
        self.assertEqual(opt_uw.shape, self.img_soft.shape)

        initial_ratio = telemetry["initial_stats"]["gradient_ratio"]
        final_ratio = telemetry["final_stats"]["gradient_ratio"]

        # Final ratio should be closer to 1.0 than initial ratio
        self.assertLess(abs(final_ratio - 1.0), abs(initial_ratio - 1.0))


if __name__ == "__main__":
    unittest.main()
