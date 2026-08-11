import os
import sys
import json
import time
import unittest
import threading
import tempfile
import urllib.request
from pathlib import Path
import numpy as np
import cv2

# Ensure repo root is in python path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.realworld.ios_bridge.snapshot_server import ThreadedHTTPServer, SnapshotRequestHandler


class TestSnapshotServerPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        calibration = {
            "K_main": [[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1]],
            "D_main": [0, 0, 0, 0, 0],
            "K_ultrawide": [[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1]],
            "D_ultrawide": [0, 0, 0, 0, 0],
            "R_main_to_ultrawide": np.eye(3).tolist(),
            "T_main_to_ultrawide_m": [0.02, 0, 0],
            "main_native_size": [640, 480],
            "ultrawide_native_size": [640, 480],
            "lens_model": "pinhole",
        }
        calibration_result = cls.root / "calibration-result.json"
        calibration_result.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "input_limited",
                    "reason_code": "TEST_CALIBRATION_LIMITATION",
                    "recommended_action": "Capture calibration evidence.",
                    "trusted_calibration": None,
                    "diagnostic_candidate": calibration,
                }
            ),
            encoding="utf-8",
        )

        class TestHandler(SnapshotRequestHandler):
            calibration_result_path = str(calibration_result)
            digest_output_dir = str(cls.root / "digest")

        cls.handler = TestHandler
        cls.test_port = 8798
        cls.server = ThreadedHTTPServer(("127.0.0.1", cls.test_port), cls.handler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.shutdown()
            cls.server.server_close()
        cls.temp.cleanup()

    def test_single_snapshot_http_post_upload(self):
        rng = np.random.default_rng(11)
        ultrawide_gray = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        ultrawide_gray = cv2.GaussianBlur(ultrawide_gray, (3, 3), 0)
        main_gray = np.zeros_like(ultrawide_gray)
        main_gray[:, :-12] = ultrawide_gray[:, 12:]
        main_img = cv2.cvtColor(main_gray, cv2.COLOR_GRAY2BGR)
        uw_img = cv2.cvtColor(ultrawide_gray, cv2.COLOR_GRAY2BGR)

        _, main_jpeg = cv2.imencode('.jpg', main_img)
        _, uw_jpeg = cv2.imencode('.jpg', uw_img)

        metadata = {
            "device": "iPhone14,2",
            "main_focal_length": 1500.0,
            "ultrawide_focal_length": 800.0,
            "baseline_m": 0.02
        }
        metadata_json = json.dumps(metadata)

        boundary = "------------------------BoundaryTest123"
        body = bytearray()

        # Part 1: main_image
        body.extend(f"--{boundary}\r\n".encode('utf-8'))
        body.extend(b'Content-Disposition: form-data; name="main_image"; filename="main.jpg"\r\n')
        body.extend(b'Content-Type: image/jpeg\r\n\r\n')
        body.extend(main_jpeg.tobytes())
        body.extend(b'\r\n')

        # Part 2: ultrawide_image
        body.extend(f"--{boundary}\r\n".encode('utf-8'))
        body.extend(b'Content-Disposition: form-data; name="ultrawide_image"; filename="ultrawide.jpg"\r\n')
        body.extend(b'Content-Type: image/jpeg\r\n\r\n')
        body.extend(uw_jpeg.tobytes())
        body.extend(b'\r\n')

        # Part 3: metadata
        body.extend(f"--{boundary}\r\n".encode('utf-8'))
        body.extend(b'Content-Disposition: form-data; name="metadata"\r\n\r\n')
        body.extend(metadata_json.encode('utf-8'))
        body.extend(b'\r\n')

        body.extend(f"--{boundary}--\r\n".encode('utf-8'))

        url = f"http://127.0.0.1:{self.test_port}/api/upload_snapshot"
        req = urllib.request.Request(url, data=bytes(body), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))

        self.assertEqual(data["status"], "diagnostic")
        self.assertIn("dashboard_url", data)
        self.assertIn("output_dir", data)
        self.assertEqual(data["source_main_url"], "/source_main.png")
        self.assertEqual(data["depth_overlay_url"], "/depth_overlay.png")
        self.assertTrue(os.path.exists(data["output_dir"]))
        depth = np.load(self.root / "digest" / "depth_m_float32.npy")
        self.assertTrue(np.isfinite(depth).any())
        self.assertGreater(float(np.nanmedian(depth)), 0.1)

    def test_portrait_auto_rotation_and_aspect_ratio_preservation(self):
        # Generate portrait Main and Ultra-Wide image buffers (H=640, W=480)
        main_img = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.rectangle(main_img, (100, 100), (300, 300), (255, 255, 255), -1)

        uw_img = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.rectangle(uw_img, (80, 100), (280, 300), (255, 255, 255), -1)

        _, main_jpeg = cv2.imencode('.jpg', main_img)
        _, uw_jpeg = cv2.imencode('.jpg', uw_img)

        boundary = "------------------------BoundaryPortraitTest456"
        body = bytearray()

        body.extend(f"--{boundary}\r\n".encode('utf-8'))
        body.extend(b'Content-Disposition: form-data; name="main_image"; filename="main.jpg"\r\n')
        body.extend(b'Content-Type: image/jpeg\r\n\r\n')
        body.extend(main_jpeg.tobytes())
        body.extend(b'\r\n')

        body.extend(f"--{boundary}\r\n".encode('utf-8'))
        body.extend(b'Content-Disposition: form-data; name="ultrawide_image"; filename="ultrawide.jpg"\r\n')
        body.extend(b'Content-Type: image/jpeg\r\n\r\n')
        body.extend(uw_jpeg.tobytes())
        body.extend(b'\r\n')

        body.extend(f"--{boundary}--\r\n".encode('utf-8'))

        url = f"http://127.0.0.1:{self.test_port}/api/upload_snapshot"
        req = urllib.request.Request(url, data=bytes(body), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))

        self.assertEqual(data["status"], "diagnostic")
        self.assertTrue(os.path.exists(data["output_dir"]))
        rectified = cv2.imread(str(self.root / "digest" / "rectified_main.png"))
        self.assertEqual(rectified.shape[:2], (480, 640))


if __name__ == "__main__":
    unittest.main()
