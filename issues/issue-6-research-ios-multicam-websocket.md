# Issue #6: Research - iOS AVCaptureMultiCamSession & WebSocket Dual-Stream Protocol

**Labels**: `wayfinder:research`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: iOS MultiCam Streaming Researcher (`81463643-1740-4ef9-8b9e-3d247753741a`)  
**Blocks**: [Issue #7](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-7-prototype-swift-app-python-receiver.md)  
**Blocked by**: None (Frontier)

## Question

What are the hardware capability constraints, AVFoundation API configuration (`AVCaptureMultiCamSession`), and binary/JSON network streaming protocol (WebSocket/HTTP) required to stream synchronized dual-camera video buffers (Main + Ultra-Wide) and per-frame dynamic `AVCameraCalibrationData` from an iPhone to a Mac Python server in real time?

### Details to investigate
1. Hardware support & resolution limits for `AVCaptureMultiCamSession.isMultiCamSupported` on modern iPhones (e.g. 1080p @ 30fps simultaneous dual stream).
2. Extraction of per-frame `AVCameraCalibrationData` (`intrinsicMatrix`, `extrinsicMatrix(from:to:)`) from `AVCaptureOutput` connections.
3. Payload schema for WebSocket transmission (JPEG/H.264 video chunks + calibration JSON).
