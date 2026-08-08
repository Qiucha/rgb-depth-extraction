"""
HTTP Server & Browser Launcher for Stereo Depth Extraction HTML Visualization Digest.
Usage:
    python3 serve_digest.py [--dir digest_live_iphone] [--port 8080] [--no-browser]
"""

import os
import sys
import time
import argparse
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler


from functools import partial


class DigestHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler that serves files from specified directory with CORS enabled."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # Silence default request logging for cleaner console output
        pass


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def serve_digest(digest_dir: str = "digest_live_iphone", port: int = 8080, open_browser: bool = True, block: bool = True):
    """
    Launches local HTTP server serving the visual digest dashboard and opens it in default browser.
    """
    abs_digest_dir = os.path.abspath(digest_dir)
    index_path = os.path.join(abs_digest_dir, "index.html")

    if not os.path.exists(index_path):
        print(f"[ServeDigest] Warning: 'index.html' not found in '{digest_dir}'.")
        print(f"[ServeDigest] Please ensure pipeline has generated the visual digest first.")
        return None

    handler_class = partial(DigestHTTPRequestHandler, directory=abs_digest_dir)
    server = None
    target_port = port

    for p in range(port, port + 10):
        try:
            server = ReusableHTTPServer(("0.0.0.0", p), handler_class)
            target_port = p
            break
        except OSError:
            continue

    if server is None:
        print(f"[ServeDigest] Error: Could not bind to any port between {port} and {port + 9}")
        return None

    url = f"http://localhost:{target_port}"

    print(f"\n========================================================")
    print(f"🚀 VISUAL DIGEST DASHBOARD SERVER ACTIVE")
    print(f"========================================================")
    print(f"  - Digest Directory: {abs_digest_dir}")
    print(f"  - Dashboard URL:    {url}")
    print(f"========================================================\n")

    if open_browser:
        def _open():
            time.sleep(0.5)
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"[ServeDigest] Error opening browser: {e}")
        threading.Thread(target=_open, daemon=True).start()

    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[ServeDigest] Server stopped.")
            server.server_close()
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Serve Stereo Depth Visual Digest Dashboard")
    parser.add_argument("--dir", default="digest_live_iphone", help="Digest directory containing index.html and data.json")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve HTTP dashboard (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    serve_digest(digest_dir=args.dir, port=args.port, open_browser=not args.no_browser, block=True)
