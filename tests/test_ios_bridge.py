"""
Unit tests for src/realworld/ios_bridge module architecture.
"""

import unittest
import struct
import json
import numpy as np
import cv2

from src.realworld.ios_bridge.server import BinaryPacketDecoder, HEADER_STRUCT, HEADER_MAGIC


class TestIOSBridge(unittest.TestCase):

    def test_binary_packet_decoding(self):
        # Create dummy RGB images
        img_main = np.zeros((480, 640, 3), dtype=np.uint8)
        img_main[:, :, 0] = 200  # Blue box
        
        img_uw = np.zeros((480, 640, 3), dtype=np.uint8)
        img_uw[:, :, 1] = 200  # Green box

        _, main_bytes = cv2.imencode('.jpg', img_main)
        _, uw_bytes = cv2.imencode('.jpg', img_uw)

        metadata = {
            "main": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0},
            "ultrawide": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
            "extrinsics_uw_to_main": {
                "rotation_matrix_3x3": np.eye(3).tolist(),
                "translation_vector_mm": [19.5, 0.0, 0.0]
            }
        }
        meta_bytes = json.dumps(metadata).encode('utf-8')

        header = struct.pack(
            HEADER_STRUCT,
            HEADER_MAGIC,
            1,  # Version
            0,  # Flags
            1042,  # Frame ID
            1723145620000000000,  # PTS
            len(meta_bytes),
            len(main_bytes),
            len(uw_bytes)
        )

        payload = header + meta_bytes + main_bytes.tobytes() + uw_bytes.tobytes()

        frame_data = BinaryPacketDecoder.decode(payload)

        self.assertEqual(frame_data["frame_id"], 1042)
        self.assertEqual(frame_data["K_main"][0, 0], 1000.0)
        self.assertEqual(frame_data["K_uw"][0, 0], 500.0)
        self.assertEqual(frame_data["T"][0], 19.5)
        self.assertEqual(frame_data["img_main"].shape, (480, 640, 3))
        self.assertEqual(frame_data["img_uw"].shape, (480, 640, 3))

    def test_telemetry_decoding(self):
        img_dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode('.jpg', img_dummy)

        metadata = {
            "main": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0},
            "ultrawide": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
            "extrinsics_uw_to_main": {
                "rotation_matrix_3x3": np.eye(3).tolist(),
                "translation_vector_mm": [19.5, 0.0, 0.0]
            },
            "telemetry": {
                "hardware_cost": 0.85,
                "is_multi_cam_supported": True
            }
        }
        meta_bytes = json.dumps(metadata).encode('utf-8')
        header = struct.pack(
            HEADER_STRUCT,
            HEADER_MAGIC,
            1, 0, 101, 1000000,
            len(meta_bytes),
            len(img_bytes),
            len(img_bytes)
        )
        payload = header + meta_bytes + img_bytes.tobytes() + img_bytes.tobytes()
        frame_data = BinaryPacketDecoder.decode(payload)

        self.assertIn("telemetry", frame_data)
        self.assertAlmostEqual(frame_data["telemetry"]["hardware_cost"], 0.85)
        self.assertTrue(frame_data["telemetry"]["is_multi_cam_supported"])

    def test_cli_subprocess_execution(self):
        import subprocess
        import sys

        # Test direct invocation of server.py --help
        proc_server = subprocess.run(
            [sys.executable, "src/realworld/ios_bridge/server.py", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc_server.returncode, 0)
        self.assertNotIn("ModuleNotFoundError", proc_server.stderr)

        # Test direct invocation of run_ios_bridge.py --help
        proc_root = subprocess.run(
            [sys.executable, "run_ios_bridge.py", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc_root.returncode, 0)
        self.assertNotIn("ModuleNotFoundError", proc_root.stderr)
        self.assertIn("iOS Dual-Camera Real-Time Receiver", proc_root.stdout)

    def test_websockets_package_available(self):
        from src.realworld.ios_bridge.server import websockets
        self.assertIsNotNone(websockets, "websockets package must be installed in the environment")


if __name__ == '__main__':
    unittest.main()
