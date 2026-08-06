"""
Integration test suite for Baseline Raw Epipolar Pipeline (zero post-processing, no occlusion check, no filtering).
"""

import os
import json
import unittest


class TestRawBaselinePipeline(unittest.TestCase):

    def test_raw_baseline_json_telemetry(self):
        json_path = 'digest/data.json'
        self.assertTrue(os.path.exists(json_path), "digest/data.json must exist")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.assertIn('scenes', data)
        self.assertGreater(len(data['scenes']), 0)

        for sc in data['scenes']:
            self.assertIn('eval_raw', sc, f"scene {sc['name']} must contain eval_raw")
            self.assertIn('scanline_raw', sc, f"scene {sc['name']} must contain scanline_raw")

            eval_raw = sc['eval_raw']
            self.assertIn('rmse', eval_raw)
            self.assertIn('mae', eval_raw)
            self.assertGreater(eval_raw['rmse'], 0.0)

    def test_raw_baseline_assets_exist(self):
        json_path = 'digest/data.json'
        with open(json_path, 'r') as f:
            data = json.load(f)

        for sc in data['scenes']:
            scene_dir = os.path.join('digest/assets', sc['name'])
            self.assertTrue(os.path.exists(scene_dir))

            self.assertTrue(os.path.exists(os.path.join(scene_dir, 'disp_raw.jpg')), "Raw baseline disparity map must exist")
            self.assertTrue(os.path.exists(os.path.join(scene_dir, 'disp_raw_intensity.jpg')), "Raw baseline intensity map must exist")


if __name__ == '__main__':
    unittest.main()
