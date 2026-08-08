# Issue #8: Task - Fix Swift AVFoundation Xcode Build Errors & Port API Calls

**Labels**: `wayfinder:task`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: None  

## Question

How should `ios/DualCamStereoCapture/` and `ios/DualCamStereoCapture/DualCamStereoCapture2/` Swift files (`MultiCamSessionManager.swift`, `WebSocketStreamer.swift`, `ContentView.swift`) be refactored to resolve Xcode compiler errors (AVFoundation port filtering syntax, `AVCaptureSynchronizedDataCollection` element lookup, CoreMedia intrinsics extraction APIs, `import simd`) so `swiftc` and `xcodebuild` compile cleanly without errors?

## Resolution

1. Updated `MultiCamSessionManager.swift` to use proper Swift 5.9 AVFoundation port filtering (`mainInput.ports.first(where: { $0.mediaType == .video })`).
2. Updated `AVCaptureDataOutputSynchronizerDelegate` data lookup to use `synchronizedDataCollection.synchronizedData(for: mainOutput) as? AVCaptureSynchronizedSampleBufferData`.
3. Updated CoreMedia intrinsic attachment retrieval to use `CMGetAttachment` and `import simd`.
4. Resolved duplicate file conflicts (`Info.plist`, `DualCamStereoCapture2App.swift`) in `DualCamStereoCapture2.xcodeproj`.
5. Verified clean compilation via `swiftc` and **BUILD SUCCEEDED** via `xcodebuild`.
