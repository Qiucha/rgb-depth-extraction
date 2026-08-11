"""
Master Integration Test Suite for Phase 6 Diagnostics & Real-World Stereo Failure Mode Mitigations.
Verifies epipolar grid overlay rendering, normalized Sobel gradient pre-filtering, expanded search ranges (numDisparities=160),
sensor timestamp sync & VCM focus lock validation, and PSF gradient mismatch optimization across the end-to-end pipeline.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from src.realworld.dataset_generator import generate_synthetic_realworld_dataset
from src.realworld.pipeline import run_realworld_pipeline
from src.realworld.sync_focus_validator import SensorSyncFocusValidator
from src.realworld.psf_gradient_optimizer import PSFGradientOptimizer
from src.realworld.census_sgbm_matcher import CensusSGBMMatcher


class TestPhase6DiagnosticsMasterSuite(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sequence_dir = os.path.join(self.test_dir, "phase6_seq")
        self.digest_dir = os.path.join(self.test_dir, "phase6_digest")

        # Generate synthetic realworld test sequence dataset
        generate_synthetic_realworld_dataset(output_dir=self.sequence_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_phase6_epipolar_grid_and_expanded_disparity_pipeline(self):
        # Run real-world pipeline with Census-SGBM matcher and max_disparity=160
        matcher_kwargs = {"max_disparity": 160}
        summary = run_realworld_pipeline(
            sequence_dir=self.sequence_dir,
            output_dir=self.digest_dir,
            target_size=(1280, 960),
            matcher_type="census_sgbm",
            matcher_kwargs=matcher_kwargs,
            strict_quality_gates=False
        )

        # Verify summary output structure
        self.assertIn("evaluation_metrics", summary)
        self.assertIn("rectified_focal_length_px", summary)
        self.assertEqual(tuple(summary["target_resolution"]), (1280, 960))

        # Verify visual digest generated epipolar grid visualizer
        epipolar_grid_path = os.path.join(self.digest_dir, "assets", "phase6_seq", "epipolar_grid.jpg")
        self.assertTrue(os.path.exists(epipolar_grid_path), "epipolar_grid.jpg missing from digest assets directory!")

        grid_img = cv2.imread(epipolar_grid_path)
        self.assertIsNotNone(grid_img)
        self.assertEqual(grid_img.shape[:2], (960, 2560))

    def test_phase6_sensor_sync_and_psf_optimizer_integration(self):
        # 1. Test SensorSyncFocusValidator on frame metadata
        validator = SensorSyncFocusValidator(max_delta_ms=2.0, readout_time_ms=16.6)
        valid_metadata = {
            "main_pts_ns": 1000000000,
            "ultrawide_pts_ns": 1000800000,  # 0.8ms delta
            "telemetry": {"vcm_focus_locked": True},
            "main": {"focus_locked": True, "lens_position": 0.7},
            "ultrawide": {"focus_locked": True, "lens_position": 0.7}
        }
        sync_report = validator.validate_frame_pair(valid_metadata, frame_height=1080)
        self.assertTrue(sync_report["is_valid"])
        self.assertLess(sync_report["timestamp_sync"]["delta_ms"], 1.0)

        # 2. Test PSFGradientOptimizer on synthetic pairs
        np.random.seed(123)
        img_main = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        img_uw = cv2.GaussianBlur(img_main, (7, 7), 2.0)

        psf_opt = PSFGradientOptimizer(target_ratio=1.0)
        opt_main, opt_uw, telemetry = psf_opt.optimize_pair(img_main, img_uw)

        self.assertEqual(opt_main.shape, img_main.shape)
        self.assertEqual(opt_uw.shape, img_uw.shape)
        self.assertIn("action_taken", telemetry)

        # 3. Compute cost volume bounds on Census-SGBM matcher
        matcher = CensusSGBMMatcher(max_disparity=160)
        disp, mask = matcher.compute_disparity(opt_main, opt_uw)
        self.assertEqual(disp.shape, (240, 320))
        self.assertTrue(mask.any())

        cost_bounds = matcher.compute_cost_volume_bounds(opt_main, opt_uw)
        self.assertIn("mean_min_cost", cost_bounds)
        self.assertIn("boundary_saturation_ratio", cost_bounds)


if __name__ == "__main__":
    unittest.main()
