"""
Unit and integration tests for tools/calibrate_offline.py CLI tool.
"""

import os
import json
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from tools.calibrate_offline import load_image_pairs, main as cli_main
from src.realworld.pipeline import run_realworld_pipeline


class TestCalibrateOfflineCLI(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        # Create a synthetic checkerboard image
        h, w = 480, 640
        self.board_img = np.ones((h, w, 3), dtype=np.uint8) * 255
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

        # Save synthetic snapshot pair files
        self.snap_dir = os.path.join(self.tmp_dir, "snapshot_001")
        os.makedirs(self.snap_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.snap_dir, "main.jpg"), self.board_img)
        cv2.imwrite(os.path.join(self.snap_dir, "ultrawide.jpg"), self.board_img)

        self.json_out = os.path.join(self.tmp_dir, "calib_out.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_image_pairs(self):
        pairs, orig_uw_sizes = load_image_pairs(self.tmp_dir)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].shape, self.board_img.shape)

    def test_cli_execution_and_json_export(self):
        # Mock sys.argv for CLI main invocation
        orig_argv = os.sys.argv
        os.sys.argv = [
            "calibrate_offline.py",
            "--images_dir", self.tmp_dir,
            "--pattern_cols", "9",
            "--pattern_rows", "6",
            "--square_size_m", "0.025",
            "--output_json", self.json_out
        ]
        try:
            cli_main()
        finally:
            os.sys.argv = orig_argv

        self.assertTrue(os.path.exists(self.json_out))
        with open(self.json_out, "r") as f:
            data = json.load(f)

        self.assertIn("K1", data)
        self.assertIn("D1", data)
        self.assertIn("K2", data)
        self.assertIn("D2", data)
        self.assertIn("R", data)
        self.assertIn("T", data)
        self.assertEqual(data["num_pairs_used"], 1)

    def test_pipeline_integration_with_exported_json(self):
        # Create a synthetic pipeline sequence folder
        seq_dir = os.path.join(self.tmp_dir, "test_seq")
        os.makedirs(seq_dir, exist_ok=True)

        manifest = {
            "version": "1.0",
            "frame_rate": 30,
            "frames": [{
                "frame_id": 0,
                "timestamp_ns": 1000000,
                "files": {
                    "main_rgb": "im0.jpg",
                    "ultrawide_rgb": "im1.jpg"
                },
                "iphone_calibration": {
                    "main_intrinsics": {
                        "matrix_3x3": [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
                    },
                    "ultrawide_intrinsics": {
                        "matrix_3x3": [[250, 0, 320], [0, 250, 240], [0, 0, 1]]
                    },
                    "extrinsic_transform_ultrawide_to_main": {
                        "rotation_matrix_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "translation_vector_mm": [[-19.5], [0], [0]]
                    }
                }
            }]
        }

        with open(os.path.join(seq_dir, "dataset_manifest.json"), "w") as f:
            json.dump(manifest, f)

        cv2.imwrite(os.path.join(seq_dir, "im0.jpg"), self.board_img)
        cv2.imwrite(os.path.join(seq_dir, "im1.jpg"), self.board_img)

        # Run pipeline with refined_calib override loaded from json
        refined_calib = {
            "K1": np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
            "D1": np.zeros(5, dtype=np.float64),
            "K2": np.array([[250, 0, 320], [0, 250, 240], [0, 0, 1]], dtype=np.float64),
            "D2": np.zeros(5, dtype=np.float64),
            "R": np.eye(3, dtype=np.float64),
            "T": np.array([[-0.0195], [0], [0]], dtype=np.float64)
        }

        output_dir = os.path.join(self.tmp_dir, "digest_out")
        summary = run_realworld_pipeline(
            sequence_dir=seq_dir,
            output_dir=output_dir,
            refined_calib=refined_calib,
            strict_quality_gates=False
        )
        self.assertIsNotNone(summary)
        self.assertTrue(os.path.exists(os.path.join(output_dir, "index.html")))


if __name__ == "__main__":
    unittest.main()
