"""
Master entry point for running the iOS Dual-Camera Single-Snapshot HTTP Upload Bridge Server.
Usage:
    python3 run_snapshot_server.py [--port 8766]
"""

import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from src.realworld.ios_bridge.snapshot_server import start_snapshot_server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="iOS Dual-Camera Single-Snapshot HTTP Upload Bridge Server")
    parser.add_argument("--port", type=int, default=8766, help="Port to listen for HTTP snapshot uploads (default: 8766)")
    args = parser.parse_args()

    start_snapshot_server(port=args.port)
