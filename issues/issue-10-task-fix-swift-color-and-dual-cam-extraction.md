# Issue #10: Task - Fix Swift AVFoundation Color Pixel Format & Dual-Camera Output Extraction

**Labels**: `wayfinder:task`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: [Issue #11](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-11-task-async-live-digest-web-server.md)  
**Blocked by**: None  

## Question

How should `MultiCamSessionManager.swift` be updated to configure 32-bit BGRA color video output settings and extract distinct, synchronized sample buffers for both Main and Ultra-Wide cameras?

## Resolution

- Configure `videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]` on both `mainOutput` and `uwOutput`.
- Extract distinct `mainData` and `uwData` sample buffers from `AVCaptureSynchronizedDataCollection`.
- Encode distinct full-color `mainJPEG` and `ultrawideJPEG` in `StereoFramePacket`.
