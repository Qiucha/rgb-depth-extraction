import os
import sys

# Ensure repository root is in python path for direct script execution
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import asyncio
import json
import struct
import time
import cv2
import numpy as np
try:
    import websockets
except ImportError:
    websockets = None
from typing import Callable, Optional, Dict, Any

from src.realworld.pipeline import run_realworld_pipeline
from src.realworld.hetero_rectifier import HeterogeneousStereoRectifier
from src.realworld.digest_builder import generate_realworld_digest
from src.stereo_depth.sliding_window import SlidingWindowMatcher
from src.stereo_depth.depth_calculator import DepthCalculator
from serve_digest import serve_digest

HEADER_MAGIC = b"ROBO"
HEADER_STRUCT = "<4sHHQQIII"  # Magic(4s), Version(H), Flags(H), FrameID(Q), PTS(Q), MetaLen(I), MainLen(I), UWLen(I)
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 36 bytes


class BinaryPacketDecoder:
    """Decodes low-latency binary multiplexed frames from iOS AVCaptureMultiCamSession."""

    @staticmethod
    def decode(payload: bytes) -> Dict[str, Any]:
        if len(payload) < HEADER_SIZE:
            raise ValueError(f"Payload size ({len(payload)}) smaller than header size ({HEADER_SIZE})")

        magic, version, flags, frame_id, pts_ns, meta_len, main_len, uw_len = struct.unpack(
            HEADER_STRUCT, payload[:HEADER_SIZE]
        )

        if magic != HEADER_MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")

        offset = HEADER_SIZE
        meta_bytes = payload[offset: offset + meta_len]
        offset += meta_len

        main_jpeg = payload[offset: offset + main_len]
        offset += main_len

        uw_jpeg = payload[offset: offset + uw_len]

        metadata = json.loads(meta_bytes.decode("utf-8"))
        img_main = cv2.imdecode(np.frombuffer(main_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_uw = cv2.imdecode(np.frombuffer(uw_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img_main is None or img_uw is None:
            raise ValueError("Corrupted or incomplete JPEG buffer in frame payload")

        K_main = np.array([
            [metadata["main"]["fx"], 0.0, metadata["main"]["cx"]],
            [0.0, metadata["main"]["fy"], metadata["main"]["cy"]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        K_uw = np.array([
            [metadata["ultrawide"]["fx"], 0.0, metadata["ultrawide"]["cx"]],
            [0.0, metadata["ultrawide"]["fy"], metadata["ultrawide"]["cy"]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        R = np.array(metadata["extrinsics_uw_to_main"]["rotation_matrix_3x3"], dtype=np.float64)
        T = np.array(metadata["extrinsics_uw_to_main"]["translation_vector_mm"], dtype=np.float64)

        telemetry = metadata.get("telemetry", {})

        return {
            "frame_id": frame_id,
            "pts_ns": pts_ns,
            "img_main": img_main,
            "img_uw": img_uw,
            "K_main": K_main,
            "K_uw": K_uw,
            "R": R,
            "T": T,
            "metadata": metadata,
            "telemetry": telemetry
        }


class IOSBridgeServer:
    """Async WebSockets server receiving live dual-stream frames from iOS."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, save_dir: str = "data/live_iphone_capture", auto_process: bool = True):
        self.host = host
        self.port = port
        self.save_dir = save_dir
        self.auto_process = auto_process
        self.frame_count = 0
        self.start_time = None
        self.captured_frames = []
        self.server_instance = None

    def _process_and_generate_live_digest(self, frame_data: Dict[str, Any], out_dir: str):
        """Runs stereo rectification, block matching, and live digest HTML update off the main event loop."""
        try:
            rectifier = HeterogeneousStereoRectifier(target_size=(640, 480))
            rect_main, rect_uw, P1, P2, Q = rectifier.rectify_pair(
                img_main=frame_data["img_main"], img_uw=frame_data["img_uw"],
                K1=frame_data["K1"], D1=np.zeros(5),
                K2=frame_data["K2"], D2=np.zeros(5),
                R=frame_data["R"], T=frame_data["T"]
            )
            gray_main = cv2.cvtColor(rect_main, cv2.COLOR_BGR2GRAY) if rect_main.ndim == 3 else rect_main
            gray_uw = cv2.cvtColor(rect_uw, cv2.COLOR_BGR2GRAY) if rect_uw.ndim == 3 else rect_uw
            matcher = SlidingWindowMatcher(window_size=7, max_disparity=48, metric="zncc")
            raw_disp, _ = matcher.compute_disparity(gray_main, gray_uw)
            f_rect = P1[0, 0]
            baseline_m = float(np.linalg.norm(frame_data["T"])) / 1000.0 if np.linalg.norm(frame_data["T"]) > 1.0 else float(np.linalg.norm(frame_data["T"]))
            depth_calc = DepthCalculator(focal_length=f_rect, baseline=baseline_m, doffs=0.0)
            depth_map = depth_calc.disparity_to_depth(raw_disp)

            generate_realworld_digest(
                rect_main=rect_main, rect_uw=rect_uw,
                disparity_map=raw_disp, depth_map_m=depth_map,
                focal_length_px=float(f_rect), baseline_m=float(baseline_m),
                output_dir=out_dir,
                scene_name="Live iPhone Stereo Stream",
                is_live=True
            )
        except Exception as err:
            print(f"[IOSBridgeServer] Live digest update warning: {err}")

    def save_frame_to_dataset(self, frame_data: Dict[str, Any]):
        """Saves frame to sequence directory structure for pipeline ingestion."""
        frame_idx = frame_data["frame_id"]
        frame_folder = os.path.join(self.save_dir, f"frame_{frame_idx:06d}")
        os.makedirs(frame_folder, exist_ok=True)

        main_path = os.path.join(frame_folder, "im0_main.png")
        uw_path = os.path.join(frame_folder, "im1_ultrawide.png")

        cv2.imwrite(main_path, frame_data["img_main"])
        cv2.imwrite(uw_path, frame_data["img_uw"])

        rel_main = f"frame_{frame_idx:06d}/im0_main.png"
        rel_uw = f"frame_{frame_idx:06d}/im1_ultrawide.png"

        frame_entry = {
            "frame_index": frame_idx,
            "timestamp_seconds": frame_data["pts_ns"] / 1e9,
            "files": {
                "main_rgb": rel_main,
                "ultrawide_rgb": rel_uw
            },
            "iphone_calibration": {
                "reference_dimensions": {
                    "width": frame_data["img_main"].shape[1],
                    "height": frame_data["img_main"].shape[0]
                },
                "main_intrinsics": {
                    "matrix_3x3": frame_data["K_main"].tolist(),
                    "pixel_size_micrometers": 1.22,
                    "lens_position": 0.5
                },
                "ultrawide_intrinsics": {
                    "matrix_3x3": frame_data["K_uw"].tolist(),
                    "pixel_size_micrometers": 1.0,
                    "lens_position": 0.0
                },
                "extrinsic_transform_ultrawide_to_main": {
                    "rotation_matrix_3x3": frame_data["R"].tolist(),
                    "translation_vector_mm": frame_data["T"].tolist()
                }
            },
            "telemetry": frame_data.get("telemetry", {})
        }
        self.captured_frames.append(frame_entry)

    def write_manifest(self):
        """Writes dataset_manifest.json upon capture completion."""
        manifest = {
            "sequence_id": "seq_live_iphone_001",
            "version": "1.0.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "devices": {
                "iphone": {
                    "model": "iPhone 15/16 Pro",
                    "os_version": "iOS 17/18",
                    "nominal_baseline_mm": 19.5,
                    "main_camera": {"sensor_name": "Main Wide", "nominal_focal_length_mm": 6.86, "nominal_fov_deg": 80.0},
                    "ultrawide_camera": {"sensor_name": "Ultra-Wide", "nominal_focal_length_mm": 2.22, "nominal_fov_deg": 120.0}
                }
            },
            "frame_count": len(self.captured_frames),
            "frames": self.captured_frames
        }
        os.makedirs(self.save_dir, exist_ok=True)
        manifest_path = os.path.join(self.save_dir, "dataset_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[IOSBridgeServer] Manifest written to {manifest_path}")
        return manifest_path

    async def _handle_connection(self, websocket):
        print(f"[IOSBridgeServer] iPhone connected from {websocket.remote_address}")
        self.start_time = time.time()
        self.frame_count = 0
        self.captured_frames = []
        out_dir = "digest_live_iphone"

        # Initialize digest folder and index.html upfront so serve_digest can launch immediately
        os.makedirs(out_dir, exist_ok=True)
        index_path = os.path.join(out_dir, "index.html")
        if not os.path.exists(index_path):
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_disp = np.zeros((480, 640), dtype=np.float32)
            dummy_depth = np.zeros((480, 640), dtype=np.float32)
            generate_realworld_digest(
                rect_main=dummy_img, rect_uw=dummy_img,
                disparity_map=dummy_disp, depth_map_m=dummy_depth,
                focal_length_px=500.0, baseline_m=0.02,
                output_dir=out_dir,
                scene_name="Live iPhone Stereo Stream",
                is_live=True
            )

        if not self.server_instance:
            self.server_instance = serve_digest(out_dir, port=8080, open_browser=True, block=False)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    frame_data = BinaryPacketDecoder.decode(message)
                    self.frame_count += 1
                    self.save_frame_to_dataset(frame_data)

                    telemetry = frame_data.get("telemetry", {})
                    hw_cost = telemetry.get("hardware_bandwidth_cost", telemetry.get("hardware_cost", 0.0))
                    is_multi_cam = telemetry.get("is_multi_cam_supported", True)

                    if not is_multi_cam:
                        print("[IOSBridgeServer] WARNING: Hardware multi-camera streaming is NOT supported on sending device (e.g. running on Xcode Simulator).")
                    if hw_cost > 1.0:
                        print(f"[IOSBridgeServer] WARNING: Hardware bandwidth cost budget exceeded (cost: {hw_cost:.2f} > 1.0)! AVFoundation frames may drop or output solid black.")

                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    if self.frame_count % 30 == 0:
                        print(f"[IOSBridgeServer] Streamed Frame #{frame_data['frame_id']} | FPS: {fps:.1f} | Hardware Bandwidth Cost: {hw_cost:.2f}")

                    # Real-Time Inspection: update depth extraction asynchronously every 15 frames (and on frame 1)
                    if self.frame_count % 15 == 0 or self.frame_count == 1:
                        asyncio.create_task(asyncio.to_thread(self._process_and_generate_live_digest, frame_data, out_dir))

        except Exception as e:
            print(f"[IOSBridgeServer] Connection closed with info: {e}")
        finally:
            print(f"\n[IOSBridgeServer] iPhone disconnected. Total frames captured: {self.frame_count}")
            if self.frame_count > 0:
                self.write_manifest()
                if self.auto_process:
                    print(f"[IOSBridgeServer] Processing final high-resolution depth extraction pipeline...")
                    await asyncio.to_thread(run_realworld_pipeline, self.save_dir, output_dir=out_dir)
                    print(f"\n[IOSBridgeServer] Final visual digest dashboard updated on http://localhost:8080")

    async def start(self):
        if websockets is None:
            print("[IOSBridgeServer] Error: 'websockets' package is not installed.")
            print("[IOSBridgeServer] Install it using 'pip install websockets' to run the live WebSocket server.")
            return

        print(f"[IOSBridgeServer] Server listening on ws://{self.host}:{self.port}...")
        async with websockets.serve(self._handle_connection, self.host, self.port, max_size=20_000_000):
            await asyncio.Future()


def start_bridge_server(port: int = 8765, save_dir: str = "data/live_captures/iphone_capture_latest"):
    """Helper CLI entry point to run server."""
    server = IOSBridgeServer(port=port, save_dir=save_dir)
    asyncio.run(server.start())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iOS Dual-Camera Real-Time Receiver Bridge Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen for iOS WebSocket client (default: 8765)")
    parser.add_argument("--save-dir", default="data/live_captures/iphone_capture_latest", help="Directory where captured live sequence frames are saved")
    args = parser.parse_args()

    start_bridge_server(port=args.port, save_dir=args.save_dir)
