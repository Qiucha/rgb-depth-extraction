"""
Unit tests for src/realworld module architecture.
"""

import unittest
import numpy as np

from src.realworld.hetero_rectifier import HeterogeneousStereoRectifier
from src.realworld.evaluator import RealWorldEvaluator
from src.realworld.realsense_icp import RealSensePointcloudAligner


class TestRealWorldPipeline(unittest.TestCase):

    def setUp(self):
        self.rectifier = HeterogeneousStereoRectifier(target_size=(640, 480))
        self.evaluator = RealWorldEvaluator()
        self.aligner = RealSensePointcloudAligner()

    def test_intrinsics_scaling(self):
        K = np.array([[1000.0, 0.0, 500.0],
                      [0.0, 1000.0, 500.0],
                      [0.0, 0.0, 1.0]])
        K_scaled = self.rectifier.scale_intrinsics(K, orig_size=(1000, 1000), target_size=(500, 500))
        self.assertEqual(K_scaled[0, 0], 500.0)
        self.assertEqual(K_scaled[1, 1], 500.0)
        self.assertEqual(K_scaled[0, 2], 250.0)
        self.assertEqual(K_scaled[1, 2], 250.0)

    def test_disparity_to_depth(self):
        disp = np.array([[10.0, 20.0], [0.0, 40.0]], dtype=np.float32)
        f_rect = 500.0
        baseline = 0.02
        depth = self.rectifier.disparity_to_depth(disp, f_rect=f_rect, baseline=baseline)
        self.assertAlmostEqual(depth[0, 0], (500.0 * 0.02) / 10.0)
        self.assertAlmostEqual(depth[0, 1], (500.0 * 0.02) / 20.0)
        self.assertEqual(depth[1, 0], 0.0)

    def test_evaluator_metrics(self):
        depth_est = np.ones((100, 100), dtype=np.float32) * 1.5
        depth_gt = np.ones((100, 100), dtype=np.float32) * 1.5
        metrics = self.evaluator.evaluate_all(depth_est, depth_gt)
        self.assertEqual(metrics["mae_m"], 0.0)
        self.assertEqual(metrics["rmse_m"], 0.0)
        self.assertEqual(metrics["bad_pixel_ratio"], 0.0)

    def test_icp_fallback_alignment(self):
        src_pts = np.random.rand(50, 3).astype(np.float32) + 1.0
        tgt_pts = src_pts + 0.1
        aligned_pts, transform_mat, fitness, rmse = self.aligner.align_icp(src_pts, tgt_pts)
        self.assertEqual(aligned_pts.shape, (50, 3))
        self.assertEqual(transform_mat.shape, (4, 4))

    def test_end_to_end_realworld_pipeline(self):
        from src.realworld.dataset_generator import generate_synthetic_realworld_dataset
        from src.realworld.pipeline import run_realworld_pipeline
        import os

        seq_dir = generate_synthetic_realworld_dataset("data/test_realworld_seq")
        results = run_realworld_pipeline(seq_dir, output_dir="digest_test_realworld", target_size=(320, 240))
        
        self.assertIn("evaluation_metrics", results)
        self.assertIn("mae_m", results["evaluation_metrics"])
        self.assertTrue(os.path.exists("digest_test_realworld/realworld_summary.json"))

    def test_end_to_end_realworld_pipeline_deep_matcher(self):
        from src.realworld.dataset_generator import generate_synthetic_realworld_dataset
        from src.realworld.pipeline import run_realworld_pipeline
        import os

        seq_dir = generate_synthetic_realworld_dataset("data/test_realworld_seq")
        results = run_realworld_pipeline(seq_dir, output_dir="digest_test_realworld_deep", target_size=(320, 240), matcher_type="cre_stereo")
        
        self.assertIn("evaluation_metrics", results)
        self.assertTrue(os.path.exists("digest_test_realworld_deep/realworld_summary.json"))


if __name__ == '__main__':
    unittest.main()
