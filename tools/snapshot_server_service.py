"""Install and inspect the persistent macOS snapshot-upload server.

The public interface is this CLI::

    python3 tools/snapshot_server_service.py install
    python3 tools/snapshot_server_service.py status
    python3 tools/snapshot_server_service.py uninstall
"""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path


LAUNCH_AGENT_LABEL = "local.roboticsneo.snapshot-server"


def build_launch_agent(repo_root: Path, python_executable: Path) -> bytes:
    """Render the launchd contract for the snapshot and digest servers."""
    runtime_dir = repo_root / ".runtime"
    payload = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [
            str(python_executable),
            str(repo_root / "run_snapshot_server.py"),
            "--port",
            "8766",
        ],
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Interactive",
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(runtime_dir / "snapshot-server.log"),
        "StandardErrorPath": str(runtime_dir / "snapshot-server.error.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _service_target() -> str:
    return f"{_domain()}/{LAUNCH_AGENT_LABEL}"


def _agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def install(repo_root: Path, python_executable: Path) -> None:
    repo_root = repo_root.resolve()
    python_executable = python_executable.resolve()
    server_entrypoint = repo_root / "run_snapshot_server.py"
    if not server_entrypoint.is_file():
        raise FileNotFoundError(f"Snapshot server entrypoint not found: {server_entrypoint}")
    if not python_executable.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")

    (repo_root / ".runtime").mkdir(parents=True, exist_ok=True)
    agent_path = _agent_path()
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = agent_path.with_suffix(".plist.tmp")
    temporary.write_bytes(build_launch_agent(repo_root, python_executable))
    temporary.replace(agent_path)

    subprocess.run(
        ["launchctl", "bootout", _service_target()],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    bootstrap = None
    for _ in range(20):
        bootstrap = subprocess.run(
            ["launchctl", "bootstrap", _domain(), str(agent_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if bootstrap.returncode == 0:
            break
        time.sleep(0.25)
    if bootstrap is None or bootstrap.returncode != 0:
        detail = (bootstrap.stderr if bootstrap else "").strip()
        raise RuntimeError(f"Could not register snapshot service: {detail}")
    subprocess.run(["launchctl", "enable", _service_target()], check=True)
    subprocess.run(
        ["launchctl", "kickstart", "-k", _service_target()], check=True
    )
    print(f"Installed and started {LAUNCH_AGENT_LABEL} on port 8766.")


def status() -> int:
    completed = subprocess.run(
        ["launchctl", "print", _service_target()], check=False
    )
    return completed.returncode


def uninstall() -> None:
    subprocess.run(
        ["launchctl", "bootout", _service_target()],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    agent_path = _agent_path()
    if agent_path.exists():
        agent_path.unlink()
    print(f"Uninstalled {LAUNCH_AGENT_LABEL}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage the persistent iPhone snapshot upload server."
    )
    parser.add_argument("command", choices=("install", "status", "uninstall"))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        dest="python_executable",
    )
    args = parser.parse_args()

    if args.command == "install":
        install(args.repo_root, args.python_executable)
        return 0
    if args.command == "status":
        return status()
    uninstall()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
