# Issue #19: Task - HTTP POST Single-Pair Snapshot Upload Server Bridge

## Question

How can we build a lightweight HTTP POST upload server in Python (`src/realworld/ios_bridge/snapshot_server.py`) to receive single-shot iPhone stereo photo pairs, execute depth extraction, and serve the visual digest dashboard?

## Context & Requirements

To ingest single snapshot pairs captured by the iPhone app:
1. Implement an HTTP POST receiver endpoint (`/api/upload_snapshot`) using `aiohttp` or Python's `http.server` / `asyncio`.
2. Accept multipart form data containing `main_image` (JPEG), `ultrawide_image` (JPEG), and `metadata` (JSON).
3. Save incoming snapshot pairs to `data/live_captures/snapshot_<timestamp>/`.
4. Automatically invoke `run_custom_iphone_depth(...)` to execute heterogeneous FOV alignment, dynamic epipolar rectification, sliding window stereo matching, WLS filtering, and metric depth calculation ($Z = \frac{f \cdot B}{d}$).
5. Update the interactive digest dashboard on port 8080 and return the dashboard JSON response with summary metrics (`min_depth_m`, `max_depth_m`, `focal_length_px`).

## Resolution

1. Created `src/realworld/ios_bridge/snapshot_server.py` and `run_snapshot_server.py` providing a multi-threaded HTTP server listening on port 8766.
2. Implemented multipart parsing using standard library `email.parser` (compatible across Python 3.8 to 3.13+).
3. Receives `main_image`, `ultrawide_image`, and `metadata`, saves them to `data/live_captures/snapshot_<timestamp>/`, triggers `run_custom_iphone_depth(...)`, and launches the interactive visual digest studio on port 8080.
4. Returns JSON response containing summary depth metrics and dashboard URL (`http://localhost:8080`).
5. Verified CLI execution (`python3 run_snapshot_server.py --help`).
