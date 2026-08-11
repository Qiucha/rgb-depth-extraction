# Issue #20: Task - iOS Snapshot Upload UI & One-Tap Depth Pipeline Integration

## Question

How can we integrate single-snapshot capture and HTTP POST upload into the iOS app's SwiftUI interface (`ContentView.swift`) to provide a seamless "Capture & Process Depth" user workflow?

## Context & Requirements

1. Add a dedicated **"Capture Single Snapshot & Compute Depth"** button in `ContentView.swift`.
2. When tapped:
   - Invoke `SingleSnapshotCaptureManager.captureSnapshot()`.
   - Show a loading indicator ("Capturing & Uploading Snapshot...").
   - Construct HTTP POST multipart upload request to `http://<serverIP>:8080/api/upload_snapshot`.
   - Send `main.jpg`, `ultrawide.jpg`, and camera metadata.
   - Parse server JSON response containing metric depth metrics (`minDepth`, `maxDepth`) and digest dashboard URL (`http://<serverIP>:8080`).
3. Display a thumbnail preview of the captured Main RGB photo alongside the processed depth map overlay.

## Resolution

1. Updated `ContentView.swift` in `ios/DualCamStereoCapture/DualCamStereoCapture2/` with a prominent **"Capture Photo Pair & Extract Depth"** UI button.
2. Integrated `SingleSnapshotCaptureManager` to capture synchronized dual-camera photo pairs on single button tap.
3. Implemented robust HTTP POST multipart payload construction sending `main_image`, `ultrawide_image`, and `metadata` to `http://<serverIP>:8766/api/upload_snapshot`.
4. Displays live progress state and presents the dashboard link (`http://<serverIP>:8080`) upon extraction completion.
5. Verified clean build with `xcodebuild` (**BUILD SUCCEEDED**).
