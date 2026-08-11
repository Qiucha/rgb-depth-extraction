# Issue #18: Task - Single-Snapshot Dual-Camera Photo Capture Swift Engine

## Question

How can we build a single-snapshot dual-camera photo capture engine in Swift (`SingleSnapshotCaptureManager.swift`) that captures synchronized high-resolution Main and Ultra-Wide photos on button tap without network streaming drops?

## Context & Requirements

Continuous video streaming over WebSockets can suffer from hardware bandwidth contention, Wi-Fi packet drops, and device thermal throttling. A single-snapshot dual-camera solution provides reliable, uncompressed, high-resolution stereo photo capture on demand:
1. Configure `AVCaptureSession` or `AVCapturePhotoOutput` for single-shot dual-camera capture.
2. Trigger synchronized capture of Main Wide (`24mm`/`26mm`) and Ultra-Wide (`13mm`/`14mm`) photos on a single user button tap.
3. Extract exact CoreMedia camera intrinsics matrix ($K_{main}$, $K_{uw}$) and EXIF metadata.
4. Save the photo pair (`main.jpg`, `ultrawide.jpg`, `metadata.json`) locally to the app's Documents folder.

## Resolution

1. Implemented `SingleSnapshotCaptureManager.swift` in `ios/DualCamStereoCapture/` providing an asynchronous `captureSnapshot(completion:)` interface.
2. Captures a single synchronized `StereoPhotoSnapshot` frame pair on demand, returning `mainJPEG`, `ultrawideJPEG`, and `metadataJSON` with exact intrinsics.
3. Automatically stops `AVCaptureMultiCamSession` upon snapshot completion to preserve battery and hardware bandwidth.
4. Verified clean build with `xcodebuild` (**BUILD SUCCEEDED**).
