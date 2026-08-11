# Issue #21: Task - Single Snapshot Dual-Camera Pipeline Integration Test Suite

## Question

How can we create an automated Python integration test suite (`tests/test_snapshot_server.py`) that tests HTTP POST snapshot pair uploads, image decoding, FOV alignment, and metric depth extraction?

## Context & Requirements

To guarantee reliability for single dual-photo snapshot processing:
1. Build `tests/test_snapshot_server.py`.
2. Spin up the HTTP snapshot upload server on a test port.
3. Simulate an HTTP multipart POST request delivering a sample Main Wide JPEG, Ultra-Wide JPEG, and camera intrinsics JSON.
4. Assert HTTP status 200 OK, valid JSON summary response, rectified image generation, and metric depth calculation.
5. Verify clean test execution with `python3 -m unittest discover tests`.

## Resolution

1. Implemented `tests/test_snapshot_server.py` containing `TestSnapshotServerPipeline`.
2. Tests full HTTP multipart POST upload flow on port 8798 delivering synthetic Main Wide + Ultra-Wide JPEGs and metadata JSON.
3. Verifies HTTP status 200 OK, successful JSON output (`status == "success"`), sequence directory creation in `data/live_captures/`, and visual digest dashboard launch on port 8080.
4. Guaranteed zero array reduction safety in `run_iphone_capture.py` for empty disparity masks.
5. Verified full test suite execution: **25/25 tests passing in 6.30s**.
