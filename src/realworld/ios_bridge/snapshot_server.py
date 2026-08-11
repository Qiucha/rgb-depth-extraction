import os
import sys
import json
import time
import re
import mimetypes
import email.parser
from urllib.parse import unquote, urlsplit
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import numpy as np
import cv2

# Ensure repository root is in python path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_iphone_capture import run_custom_iphone_depth


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP Server for concurrent snapshot uploads."""
    daemon_threads = True


class SnapshotRequestHandler(BaseHTTPRequestHandler):

    calibration_result_path = os.environ.get(
        "IPHONE_CALIBRATION_RESULT", "calibration-runs/current/result.json"
    )
    digest_output_dir = os.environ.get(
        "IPHONE_DIGEST_OUTPUT", "digest_live_iphone"
    )

    def do_GET(self):
        request_path = unquote(urlsplit(self.path).path)
        if request_path == "/api/health":
            digest_ready = os.path.isfile(
                os.path.join(self.digest_output_dir, "index.html")
            )
            self._send_json_response(
                200,
                {
                    "status": "ready" if digest_ready else "waiting_for_capture",
                    "capture_endpoint": "/api/upload_snapshot",
                    "digest_url": "/",
                },
            )
            return

        relative_path = "index.html" if request_path == "/" else request_path.lstrip("/")
        digest_root = os.path.realpath(self.digest_output_dir)
        candidate = os.path.realpath(os.path.join(digest_root, relative_path))
        if os.path.commonpath((digest_root, candidate)) != digest_root:
            self._send_json_response(404, {"error": "Digest artifact not found"})
            return
        if not os.path.isfile(candidate):
            if request_path == "/":
                payload = (
                    "<!doctype html><meta name=viewport content='width=device-width'>"
                    "<title>iPhone Stereo Depth</title><h1>Ready for capture</h1>"
                    "<p>Open the iPhone app and tap Capture Photo Pair & Extract Depth.</p>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self._send_json_response(404, {"error": "Digest artifact not found"})
            return

        with open(candidate, "rb") as artifact:
            payload = artifact.read()
        content_type = mimetypes.guess_type(candidate)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path == "/api/upload_snapshot":
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._send_json_response(400, {"error": "Expected multipart/form-data"})
                    return

                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length)

                # Parse multipart using standard library email.parser
                msg_raw = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + body_bytes
                msg = email.parser.BytesParser().parsebytes(msg_raw)

                parts = {}
                if msg.is_multipart():
                    for part in msg.walk():
                        disposition = part.get("Content-Disposition", "")
                        if "name=" in disposition:
                            match = re.search(r'name="([^"]+)"', disposition)
                            if match:
                                field_name = match.group(1)
                                parts[field_name] = part.get_payload(decode=True)

                if "main_image" not in parts or "ultrawide_image" not in parts:
                    self._send_json_response(400, {"error": "Missing 'main_image' or 'ultrawide_image' in multipart payload"})
                    return

                main_bytes = parts["main_image"]
                uw_bytes = parts["ultrawide_image"]

                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                save_dir = os.path.join("data", "live_captures", f"snapshot_{timestamp_str}")
                os.makedirs(save_dir, exist_ok=True)

                main_path = os.path.join(save_dir, "main.jpg")
                uw_path = os.path.join(save_dir, "ultrawide.jpg")

                with open(main_path, "wb") as f:
                    f.write(main_bytes)

                with open(uw_path, "wb") as f:
                    f.write(uw_bytes)

                metadata_dict = {}
                if "metadata" in parts:
                    try:
                        metadata_dict = json.loads(parts["metadata"].decode("utf-8"))
                    except Exception:
                        pass

                with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                    json.dump(metadata_dict, f, indent=2)

                out_dir = self.digest_output_dir
                run_custom_iphone_depth(
                    main_path,
                    uw_path,
                    output_dir=out_dir,
                    calibration_result_path=self.calibration_result_path,
                )

                summary_path = os.path.join(out_dir, "summary.json")
                summary = {}
                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary = json.load(f)

                response_payload = {
                    "status": summary.get("status", "diagnostic"),
                    "message": (
                        "Single dual-camera snapshot processed; open the Digest for "
                        "confidence and calibration evidence."
                    ),
                    "dashboard_url": "/",
                    "source_main_url": "/source_main.png",
                    "depth_overlay_url": "/depth_overlay.png",
                    "output_dir": out_dir,
                    "trusted_depth_eligible": summary.get(
                        "trusted_depth_eligible", False
                    ),
                    "reason_codes": summary.get("reason_codes", []),
                    "summary": summary
                }
                self._send_json_response(200, response_payload)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json_response(500, {"error": str(e)})
        else:
            self._send_json_response(404, {"error": "Endpoint not found"})

    def _send_json_response(self, status_code: int, payload: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def log_message(self, format, *args):
        # Silence default HTTP access logs for clean CLI output
        pass


def start_snapshot_server(port: int = 8766):
    """Starts the single snapshot HTTP upload server."""
    server = ThreadedHTTPServer(("0.0.0.0", port), SnapshotRequestHandler)
    print(f"=== Single-Snapshot Dual-Camera HTTP Upload Server active on http://0.0.0.0:{port} ===")
    print(f"POST single photo pairs to http://<server_ip>:{port}/api/upload_snapshot")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Single-Snapshot Dual-Camera HTTP POST Bridge Server")
    parser.add_argument("--port", type=int, default=8766, help="Port to listen for HTTP snapshot uploads (default: 8766)")
    args = parser.parse_args()

    start_snapshot_server(port=args.port)
