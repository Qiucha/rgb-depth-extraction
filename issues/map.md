# Wayfinder Map: iPhone Dual-Camera Real-World Stereo Depth Pipeline

## Destination

A complete Python real-world pipeline (`src/realworld/`) building upon `src/stereo_depth/` to ingest heterogeneous iPhone dual-camera streams (Main + Ultra-Wide), execute FOV alignment and dynamic epipolar rectification (`cv2.stereoRectify`), compute disparity and metric depth ($Z = \frac{f \cdot B}{d}$), align 3D point clouds with Intel RealSense ground truth using Open3D ICP, evaluate performance (MAE/RMSE, texture dependency, flying pixel edge artifacts), present telemetry in the interactive digest dashboard, and receive live dual video streams from an iPhone via a Swift `AVCaptureMultiCamSession` bridge.

## Notes

- **Domain**: Computer Vision, Heterogeneous Passive Stereo Matching, Dynamic Camera Calibration, Point Cloud Alignment, Robotic Perception Benchmarking, iOS AVFoundation MultiCam Streaming.
- **Skills to Consult**: `diagnosing-bugs`, `codebase-design`, `prototype`, `grilling`, `domain-modeling`, `research`.
- **Standing Preferences**: Use existing `src/stereo_depth/` algorithm modules (`SlidingWindowMatcher`, `WLSFilter`, `DepthCalculator`, `DepthEvaluator`, `DigestGenerator`) as the initial baseline matching engine before integrating deep learning models.
- **Tracker**: Local Markdown Tracker in `issues/`.

## Decisions so far

<!-- the index — one line per closed ticket: enough to judge relevance, then zoom the link for the detail the ticket holds -->

- [Issue #1: Research - Heterogeneous FOV Alignment & Dynamic Rectification Math](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-1-research-hetero-fov-alignment.md) — Unified single-pass `cv2.stereoRectify` + `cv2.remap` achieves undistortion, FOV crop, alignment, and scale without pre-crop interpolation blur; $Z = \frac{f \cdot B}{d}$ derived & verified scale-invariant.
- [Issue #2: Research - iPhone Dual-Cam & RealSense Benchmark Dataset Schema](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-2-research-dataset-schema-format.md) — Defined dual storage architecture (unpacked sequence folder hierarchy with JSON manifest + unified HDF5 container) with `calib.txt` adapter for `src/stereo_depth/` backward compatibility.
- [Issue #3: Prototype - RealWorld Pipeline Architecture Extending stereo_depth](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-3-prototype-realworld-pipeline-architecture.md) — Prototyped complete package in `src/realworld/` (`hetero_rectifier.py`, `dataset_loader.py`, `realsense_icp.py`, `evaluator.py`, `pipeline.py`) extending `src/stereo_depth/` and verified with unit tests.
- [Issue #4: Grilling - Open3D ICP Point Cloud Alignment & Evaluation Strategy](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-4-grilling-icp-alignment-eval-strategy.md) — Locked Point-to-Point ICP (0.05m max correspondence), 4-zone distance plane binning (0.5m, 1.0m, 1.5m, 2.0m), and multi-axis evaluation suite (MAE, RMSE, Bad-Pixels >5cm, Texture Ratio, Flying Pixels).
- [Issue #5: Task - Synthetic Heterogeneous Dual-Cam & RealSense Benchmark Test Data Generator](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-5-task-synthetic-dataset-generator.md) — Implemented `src/realworld/dataset_generator.py` generating synthetic sequence datasets in `data/realworld_sample/` and verified with integration tests.
- [Issue #6: Research - iOS AVCaptureMultiCamSession & WebSocket Dual-Stream Protocol](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-6-research-ios-multicam-websocket.md) — Derived `AVCaptureMultiCamSession` format filtering (`isMultiCamSupported`), per-frame `AVCameraCalibrationData` extraction, and 36-byte header multiplexed binary wire protocol.
- [Issue #7: Prototype - Swift Dual-Camera Capture App & Python Receiver Server](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-7-prototype-swift-app-python-receiver.md) — Implemented `src/realworld/ios_bridge/server.py` WebSocket bridge and Swift AVFoundation client in `ios/DualCamStereoCapture/`, verified with unit tests.
- [Issue #8: Task - Fix Swift AVFoundation Xcode Build Errors & Port API Calls](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-8-task-fix-swift-xcode-build-errors.md) — Fixed AVFoundation port filtering syntax, `AVCaptureSynchronizedDataCollection` lookup, CoreMedia intrinsics `CMGetAttachment` extraction, `import simd`, and duplicate project references; verified with `swiftc` and `xcodebuild` (**BUILD SUCCEEDED**).
- [Issue #9: Task - Deep Disparity Model Integration Adapter Interface](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-9-task-deep-disparity-model-adapter.md) — Abstract `BaseStereoMatcher` adapter interface supporting CREStereo / AnyStereo / RAFT-Stereo deep models with graceful fallback and pipeline integration; verified with unit tests.
- [Issue #10: Task - Fix Swift AVFoundation Color Pixel Format & Dual-Camera Output Extraction](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-10-task-fix-swift-color-and-dual-cam-extraction.md) — Configured `kCVPixelFormatType_32BGRA`, extracted distinct Main and Ultra-Wide camera buffers, and enabled full-color JPEG encoding; verified with `xcodebuild`.
- [Issue #11: Task - Asynchronous Live Visual Digest Streaming & Immediate Web Server Launch](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-11-task-async-live-digest-web-server.md) — Immediate digest server launch on port 8080 and non-blocking `asyncio.to_thread` live depth extraction; verified with unit tests.
- [Issue #12: Task - Clean Up Test Output Folders & Organize Live Capture Hierarchy](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-12-task-cleanup-test-folders-and-organize-capture-hierarchy.md) — Created `data/live_captures/` storage hierarchy and cleaned up temporary test directories.
- [Issue #13: Task - Fix iPhone Picture Capture & CoreImage JPEG Encoding](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-13-task-fix-iphone-picture-capture-and-jpeg-encoding.md) — Reusable `CIContext` and automatic pixel format inference for robust picture frame capture; verified with `xcodebuild`.

## Not yet specified

- None

## Out of scope

- Real-time video streaming on embedded iOS hardware during Python execution.
- Active IR hardware projector modification or custom physical hardware rig manufacturing.

## Tickets

### Frontier (Unblocked)
- None (All tickets resolved! The route to destination is clear.)

### Blocked
- None
