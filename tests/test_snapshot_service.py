"""Contract tests for the persistent macOS snapshot-server launcher."""

import plistlib
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from src.realworld.ios_bridge.snapshot_server import (
    SnapshotRequestHandler,
    ThreadedHTTPServer,
)
from tools.snapshot_server_service import (
    LAUNCH_AGENT_LABEL,
    build_launch_agent,
)


class TestSnapshotServerService(unittest.TestCase):
    def test_launch_agent_is_login_started_and_self_restarting_on_port_8766(self):
        with tempfile.TemporaryDirectory() as temp:
            repo_root = Path(temp) / "robotics_neo"
            python = Path("/opt/homebrew/bin/python3")

            rendered = build_launch_agent(repo_root, python)
            payload = plistlib.loads(rendered)

        self.assertEqual(payload["Label"], LAUNCH_AGENT_LABEL)
        self.assertTrue(payload["RunAtLoad"])
        self.assertTrue(payload["KeepAlive"])
        self.assertEqual(payload["WorkingDirectory"], str(repo_root))
        self.assertEqual(
            payload["ProgramArguments"],
            [
                str(python),
                str(repo_root / "run_snapshot_server.py"),
                "--port",
                "8766",
            ],
        )
        self.assertEqual(
            payload["EnvironmentVariables"]["PYTHONUNBUFFERED"], "1"
        )

    def test_one_server_address_exposes_health_and_latest_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            digest = Path(temp)
            (digest / "index.html").write_text(
                "<h1>Actual camera captures</h1>", encoding="utf-8"
            )

            class TestHandler(SnapshotRequestHandler):
                digest_output_dir = str(digest)

            server = ThreadedHTTPServer(("127.0.0.1", 0), TestHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(f"{origin}/") as response:
                    page = response.read().decode("utf-8")
                with urllib.request.urlopen(f"{origin}/api/health") as response:
                    health = response.read().decode("utf-8")
            finally:
                server.shutdown()
                server.server_close()

        self.assertIn("Actual camera captures", page)
        self.assertIn('"status": "ready"', health)
        self.assertIn('"digest_url": "/"', health)


if __name__ == "__main__":
    unittest.main()
