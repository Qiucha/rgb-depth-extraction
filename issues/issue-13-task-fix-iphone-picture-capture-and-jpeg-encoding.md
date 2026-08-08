# Issue #13: Task - Fix iPhone Picture Capture & CoreImage JPEG Encoding

**Labels**: `wayfinder:task`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: None  

## Question

Why was the Swift iOS app failing to capture actual picture frames (returning nil or dropping frames during buffer conversion), and how should `MultiCamSessionManager.swift` be fixed to guarantee robust frame capture and JPEG encoding?

## Resolution

- Instantiate reusable `CIContext` instance to prevent per-frame GPU context allocation overhead.
- Remove restrictive `format: .RGBA8` parameter in `ciContext.createCGImage` to let CoreImage infer the pixel buffer format automatically from `CVPixelBuffer`.
- Add independent output fallbacks for Main and Ultra-Wide camera outputs to prevent frame drops caused by strict synchronizer timing misalignments.
