"""
Master entry point for running the iOS Dual-Camera Live WebSocket Server Bridge.
Usage:
    python3 run_ios_bridge.py [--port 8765] [--save-dir data/live_iphone_capture]
"""

import os
import sys
import argparse

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath('.'))

from src.realworld.ios_bridge.server import start_bridge_server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="iOS Dual-Camera Real-Time Receiver Bridge Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen for iOS WebSocket client (default: 8765)")
    parser.add_argument("--save-dir", default="data/live_iphone_capture", help="Directory where captured live sequence frames are saved")
    args = parser.parse_args()

    print(f"=== Starting iOS Dual-Camera Bridge Server on port {args.port} ===")
    print(f"Captured live frames will be saved to '{args.save_dir}/'")
    start_bridge_server(port=args.port, save_dir=args.save_dir)
