"""
Unit test suite for SensorSyncFocusValidator (Issue #42).
Verifies inter-sensor timestamp synchronization, rolling shutter readout drift delta_y(t),
and VCM focus lock state validation.
"""

import unittest
from src.realworld.sync_focus_validator import SensorSyncFocusValidator


class TestSensorSyncFocusValidator(unittest.TestCase):

    def setUp(self):
        self.validator = SensorSyncFocusValidator(max_delta_ms=2.0, readout_time_ms=16.6)

    def test_timestamp_sync_valid(self):
        # 1ms difference between Main and Ultra-Wide presentation timestamps
        pts_main = 1000000000
        pts_uw   = 1001000000  # +1,000,000 ns = 1.0 ms

        report = self.validator.validate_timestamps(pts_main, pts_uw, frame_height=1080)
        self.assertTrue(report["is_synced"])
        self.assertEqual(report["delta_ms"], 1.0)
        # 1080 * (1.0 / 16.6) = 65.0602 px
        self.assertAlmostEqual(report["row_drift_px"], 65.0602, places=3)

    def test_timestamp_sync_invalid_exceeds_threshold(self):
        # 5ms difference (exceeds 2ms limit)
        pts_main = 1000000000
        pts_uw   = 1005000000  # +5,000,000 ns = 5.0 ms

        report = self.validator.validate_timestamps(pts_main, pts_uw, frame_height=1080)
        self.assertFalse(report["is_synced"])
        self.assertEqual(report["delta_ms"], 5.0)

    def test_vcm_focus_lock_valid(self):
        metadata = {
            "telemetry": {"vcm_focus_locked": True},
            "main": {"focus_locked": True, "lens_position": 0.7},
            "ultrawide": {"focus_locked": True, "lens_position": 0.7}
        }

        report = self.validator.validate_vcm_focus_lock(metadata)
        self.assertTrue(report["is_focus_locked"])
        self.assertEqual(report["main_lens_position"], 0.7)
        self.assertEqual(report["ultrawide_lens_position"], 0.7)

    def test_vcm_focus_lock_invalid_hunting_or_unlocked(self):
        # Lens position drifted or unlocked
        metadata = {
            "telemetry": {"vcm_focus_locked": False},
            "main": {"focus_locked": False, "lens_position": 0.3},
            "ultrawide": {"focus_locked": True, "lens_position": 0.7}
        }

        report = self.validator.validate_vcm_focus_lock(metadata)
        self.assertFalse(report["is_focus_locked"])

    def test_validate_frame_pair_master(self):
        valid_metadata = {
            "main_pts_ns": 1000000000,
            "ultrawide_pts_ns": 1000500000, # 0.5 ms delta
            "telemetry": {"vcm_focus_locked": True},
            "main": {"focus_locked": True, "lens_position": 0.7},
            "ultrawide": {"focus_locked": True, "lens_position": 0.7}
        }

        report = self.validator.validate_frame_pair(valid_metadata, frame_height=1080)
        self.assertTrue(report["is_valid"])
        self.assertTrue(report["timestamp_sync"]["is_synced"])
        self.assertTrue(report["vcm_focus_lock"]["is_focus_locked"])


if __name__ == "__main__":
    unittest.main()
