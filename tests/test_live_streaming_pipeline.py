"""
Integration test suite for iOS Dual-Camera WebSocket Live Streaming Pipeline.
"""

import os
import sys
import time
import json
import struct
import shutil
import tempfile
import unittest
import asyncio
import numpy as np
import cv2
import websockets

from src.realworld.ios_bridge.server import IOSBridgeServer, HEADER_STRUCT, HEADER_MAGIC


def create_dummy_binary_packet(frame_id: int, pts_ns: int) -> bytes:
    """Helper to create a valid 36-byte header binary packet with dummy images."""
    img_main = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img_main, (100, 100), (300, 300), (255, 0, 0), -1)

    img_uw = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img_uw, (150, 100), (350, 300), (0, 255, 0), -1)

    _, main_bytes = cv2.imencode('.jpg', img_main)
    _, uw_bytes = cv2.imencode('.jpg', img_uw)

    metadata = {
        "frame_id": frame_id,
        "timestamp_pts_ns": pts_ns,
        "main": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0},
        "ultrawide": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        "extrinsics_uw_to_main": {
            "rotation_matrix_3x3": np.eye(3).tolist(),
            "translation_vector_mm": [19.5, 0.0, 0.0]
        },
        "telemetry": {
            "hardware_bandwidth_cost": 0.55,
            "is_multi_cam_supported": True
        }
    }
    meta_bytes = json.dumps(metadata).encode('utf-8')

    header = struct.pack(
        HEADER_STRUCT,
        HEADER_MAGIC,
        1,  # Version
        0,  # Flags
        frame_id,
        pts_ns,
        len(meta_bytes),
        len(main_bytes),
        len(uw_bytes)
    )

    return header + meta_bytes + main_bytes.tobytes() + uw_bytes.tobytes()


class TestLiveStreamingPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_live_capture_")
        self.port = 8769

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_live_websocket_streaming_and_manifest(self):
        async def run_server_and_client():
            server = IOSBridgeServer(host="127.0.0.1", port=self.port, save_dir=self.test_dir, auto_process=False)
            
            # Start server task
            server_task = asyncio.create_task(server.start())
            await asyncio.sleep(0.2)  # Give server time to bind

            uri = f"ws://127.0.0.1:{self.port}"
            async with websockets.connect(uri) as websocket:
                # Stream 3 binary frames
                for i in range(1, 4):
                    packet = create_dummy_binary_packet(frame_id=i, pts_ns=1000000000 * i)
                    await websocket.send(packet)
                    await asyncio.sleep(0.1)

            # Wait briefly for server disconnect cleanup
            await asyncio.sleep(0.3)
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

            return server.frame_count

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            frames_processed = loop.run_until_complete(run_server_and_client())
        finally:
            loop.close()

        self.assertEqual(frames_processed, 3)

        # Check saved sequence hierarchy
        manifest_path = os.path.join(self.test_dir, "dataset_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "dataset_manifest.json must be written on client disconnect")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["frame_count"], 3)
        self.assertEqual(len(manifest["frames"]), 3)

        # Verify frame files on disk
        frame_0_main = os.path.join(self.test_dir, manifest["frames"][0]["files"]["main_rgb"])
        frame_0_uw = os.path.join(self.test_dir, manifest["frames"][0]["files"]["ultrawide_rgb"])
        self.assertTrue(os.path.exists(frame_0_main))
        self.assertTrue(os.path.exists(frame_0_uw))


if __name__ == '__main__':
    unittest.main()
