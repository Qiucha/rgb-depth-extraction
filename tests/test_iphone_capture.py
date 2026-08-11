"""
Unit test suite for iPhone Capture Script with Extrinsic Matrix Calibration & Accelerometer Tilt Alignment.
"""

import os
import json
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from run_iphone_capture import run_custom_iphone_depth


class TestIPhoneCaptureExtrinsics(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.main_path = os.path.join(self.test_dir, "main.jpg")
        self.uw_path = os.path.join(self.test_dir, "ultrawide.jpg")
        self.calib_path = os.path.join(self.test_dir, "calib_custom.json")
        self.calibration_result_path = os.path.join(
            self.test_dir, "calibration-result.json"
        )
        self.out_dir = os.path.join(self.test_dir, "digest_out")

        # Synthesize synthetic Main and Ultra-Wide images
        img_main = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        img_uw = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

        cv2.imwrite(self.main_path, img_main)
        cv2.imwrite(self.uw_path, img_uw)

        # Synthesize custom extrinsic & intrinsic calibration JSON
        calib_data = {
            "K1": [[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]],
            "D1": [0.0, 0.0, 0.0, 0.0, 0.0],
            "K2": [[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]],
            "D2": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R": [[0.9998, -0.0175, 0.0], [0.0175, 0.9998, 0.0], [0.0, 0.0, 1.0]],
            "T": [19.5, 0.0, 0.0]
        }
        with open(self.calib_path, "w") as f:
            json.dump(calib_data, f, indent=2)

        calibration_result = {
            "schema_version": 1,
            "status": "input_limited",
            "reason_code": "TEST_INPUT_LIMITED",
            "recommended_action": "Capture better calibration evidence.",
            "trusted_calibration": None,
            "diagnostic_candidate": {
                "K_main": calib_data["K1"],
                "D_main": calib_data["D1"],
                "K_ultrawide": calib_data["K2"],
                "D_ultrawide": calib_data["D2"],
                "R_main_to_ultrawide": np.eye(3).tolist(),
                "T_main_to_ultrawide_m": [0.0195, 0.0, 0.0],
                "main_native_size": [640, 480],
                "ultrawide_native_size": [640, 480],
                "lens_model": "pinhole",
            },
        }
        with open(self.calibration_result_path, "w") as f:
            json.dump(calibration_result, f, indent=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_custom_iphone_depth_with_extrinsics(self):
        out = run_custom_iphone_depth(
            main_path=self.main_path,
            ultrawide_path=self.uw_path,
            output_dir=self.out_dir,
            target_size=(320, 240),
            calibration_result_path=self.calibration_result_path,
        )

        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "summary.json")))

        with open(os.path.join(self.out_dir, "summary.json"), "r") as f:
            summary = json.load(f)

        self.assertEqual(summary["status"], "diagnostic")
        self.assertFalse(summary["trusted_depth_eligible"])
        self.assertIn("TEST_INPUT_LIMITED", summary["reason_codes"])
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "stereo_result.json")))

    def test_matrix_only_calibration_is_not_auto_discovered(self):
        # Adjacent legacy matrices must not silently become trusted calibration.
        adjacent_calib = os.path.join(os.path.dirname(self.main_path), "calibration_refined.json")
        shutil.copyfile(self.calib_path, adjacent_calib)
        with self.assertRaises(ValueError):
            run_custom_iphone_depth(
                main_path=self.main_path,
                ultrawide_path=self.uw_path,
                output_dir=self.out_dir + "_legacy",
                target_size=(320, 240),
                calibration_result_path=adjacent_calib,
            )


if __name__ == "__main__":
    unittest.main()
