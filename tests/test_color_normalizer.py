"""
Unit tests for ColorExposureNormalizer preprocessing module.
"""

import unittest
import numpy as np
import cv2

from src.realworld.color_normalizer import ColorExposureNormalizer


class TestColorExposureNormalizer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create synthetic reference image (100x100 RGB)
        self.ref_img = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
        # Create target image with deliberate brightness/contrast shift
        self.target_img = (self.ref_img.astype(np.float32) * 0.4 + 40).astype(np.uint8)

    def test_luminance_matching_mean_std(self):
        normalizer = ColorExposureNormalizer(method="mean_std")
        matched = normalizer.match_luminance(self.target_img, self.ref_img)

        self.assertEqual(matched.shape, self.target_img.shape)
        self.assertEqual(matched.dtype, np.uint8)

        ref_mean = np.mean(cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2YCrCb)[:, :, 0])
        matched_mean = np.mean(cv2.cvtColor(matched, cv2.COLOR_BGR2YCrCb)[:, :, 0])
        self.assertAlmostEqual(ref_mean, matched_mean, delta=3.0)

    def test_luminance_matching_cdf(self):
        normalizer = ColorExposureNormalizer(method="cdf")
        matched = normalizer.match_luminance(self.target_img, self.ref_img)

        self.assertEqual(matched.shape, self.target_img.shape)
        self.assertEqual(matched.dtype, np.uint8)

        ref_mean = np.mean(self.ref_img)
        matched_mean = np.mean(matched)
        self.assertAlmostEqual(ref_mean, matched_mean, delta=5.0)

    def test_census_transform_5x5(self):
        gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        census = ColorExposureNormalizer.compute_census_transform_5x5(gray)

        self.assertEqual(census.shape, gray.shape)
        self.assertEqual(census.dtype, np.uint32)

    def test_census_cost_volume_zero_disparity(self):
        gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        census_l = ColorExposureNormalizer.compute_census_transform_5x5(gray)
        census_r = census_l.copy()

        cost_vol = ColorExposureNormalizer.compute_census_cost_volume(census_l, census_r, max_disparity=10)

        self.assertEqual(cost_vol.shape, (100, 100, 10))
        # At d=0, cost should be zero in interior (excluding 5x5 borders)
        interior = cost_vol[3:-3, 3:-3, 0]
        np.testing.assert_array_equal(interior, 0.0)


if __name__ == "__main__":
    unittest.main()
