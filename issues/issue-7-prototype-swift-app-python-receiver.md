# Issue #7: Prototype - Swift Dual-Camera Capture App & Python Receiver Server

**Labels**: `wayfinder:prototype`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: [Issue #6](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-6-research-ios-multicam-websocket.md)

## Question

How should the iOS Swift dual-camera capture app (`ios/DualCamStereoCapture`) and the Python receiver bridge server (`src/realworld/ios_bridge/server.py`) be implemented to receive live dual-stream frames and calibration data, saving them directly into the dataset format ingested by `run_realworld_pipeline()`?

## Resolution

1. Implemented `src/realworld/ios_bridge/server.py` async WebSocket server decoding the 36-byte header binary protocol, writing frame datasets to `data/live_iphone_capture/`, and triggering `run_realworld_pipeline()`.
2. Verified binary packet decoding via `tests/test_ios_bridge.py`.
3. Created Swift AVFoundation multi-cam client package in `ios/DualCamStereoCapture/` (`MultiCamSessionManager.swift`, `WebSocketStreamer.swift`, `ContentView.swift`).

