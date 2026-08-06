"""
Integration test suite for Disparity Intensity Panel assets and telemetry in digest/.
"""

import os
import json
import unittest


class TestDisparityIntensityPanel(unittest.TestCase):

    def test_digest_data_json_structure(self):
        json_path = 'digest/data.json'
        self.assertTrue(os.path.exists(json_path), "digest/data.json must exist")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.assertIn('scenes', data, "data.json must contain 'scenes' list")
        self.assertGreater(len(data['scenes']), 0, "scenes list must not be empty")

        for sc in data['scenes']:
            self.assertIn('name', sc)
            self.assertIn('stats_a', sc, f"scene {sc['name']} must contain stats_a")
            self.assertIn('stats_c', sc, f"scene {sc['name']} must contain stats_c")

            stats_c = sc['stats_c']
            self.assertIn('min_intensity', stats_c)
            self.assertIn('max_intensity', stats_c)
            self.assertIn('mean_intensity', stats_c)
            self.assertIn('histogram', stats_c)
            self.assertEqual(len(stats_c['histogram']), 16)

    def test_digest_assets_exist(self):
        json_path = 'digest/data.json'
        with open(json_path, 'r') as f:
            data = json.load(f)

        for sc in data['scenes']:
            scene_dir = os.path.join('digest/assets', sc['name'])
            self.assertTrue(os.path.exists(scene_dir), f"Asset directory {scene_dir} must exist")

            self.assertTrue(os.path.exists(os.path.join(scene_dir, 'disp_intensity_panel.jpg')), "4-up panel image must exist")
            self.assertTrue(os.path.exists(os.path.join(scene_dir, 'disp_c_intensity.jpg')), "Pipeline C intensity map must exist")
            self.assertTrue(os.path.exists(os.path.join(scene_dir, 'disp_a_intensity.jpg')), "Pipeline A intensity map must exist")


if __name__ == '__main__':
    unittest.main()
