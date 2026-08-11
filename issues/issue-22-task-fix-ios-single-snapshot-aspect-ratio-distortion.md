# Issue #22: Task - Fix iOS Single-Snapshot Aspect Ratio Distortion & Intrinsics Alignment

## Question

How can we eliminate aspect ratio distortion, non-uniform scaling, and optical skew in single-frame captures taken from the iOS dual-camera app, ensuring accurate stereo alignment, undistorted rectification, and robust metric depth extraction?

## Context & Requirements

1. **iOS MultiCam Output Orientation & Sensor Alignment**:
   - In `MultiCamSessionManager.swift`, setting `videoOrientation = .portrait` forced 1080x1920 portrait pixel buffers while passing landscape sensor intrinsics ($cx \approx 960, cy \approx 540$) and expecting a horizontal baseline ($T_x = 19.5\text{mm}$).
   - Update `MultiCamSessionManager.swift` to align output video connections to `.landscapeRight` / native landscape orientation, matching the physical horizontal hardware baseline and AVFoundation intrinsic matrices.

2. **Single-Snapshot Auto-Exposure & Warmup Stabilization**:
   - Update `SingleSnapshotCaptureManager.swift` / `MultiCamSessionManager.swift` to skip the initial 5-8 warmup frames after session launch before delivering the snapshot pair, allowing hardware auto-exposure (AE), auto-focus (AF), and camera frame synchronization to lock.

3. **Python Rectification & Aspect Ratio Preservation**:
   - In `src/realworld/hetero_rectifier.py` and `run_iphone_capture.py`, auto-detect image aspect ratio and orientation.
   - Ensure isotropic intrinsic scaling (`sx == sy`) and correct aspect-ratio target size selection (e.g. `(1280, 720)` for 16:9 or `(1280, 960)` for 4:3 landscape) to prevent non-uniform squishing or stretching during `cv2.remap()`.
   - If portrait images are received, auto-rotate 90 degrees counter-clockwise to landscape and adjust intrinsics $K$ before rectification.

4. **Integration Testing & Verification**:
   - Verify Swift iOS capture logic and Xcode build (`xcodebuild`).
   - Run end-to-end unit and integration test suite (`python3 -m unittest discover -s tests`).

## Resolution

1. **iOS MultiCam Connection Alignment**: Updated `MultiCamSessionManager.swift` across all target variants to output `.landscapeRight` pixel buffers, matching native camera sensor orientation, horizontal baseline vector ($T_x = 19.5\text{mm}$), and AVFoundation camera intrinsic matrices ($cx \approx 960, cy \approx 540$).
2. **Auto-Exposure & Warmup Frame Skipping**: Updated `SingleSnapshotCaptureManager.swift` to skip the initial 5 warmup frames upon session startup, allowing camera hardware auto-exposure (AE), auto-focus (AF), and auto-white-balance (AWB) to lock before snatching the single-shot photo pair.
3. **Aspect Ratio Preservation & Auto-Rotation**: Updated `HeterogeneousStereoRectifier` (`src/realworld/hetero_rectifier.py`) to auto-detect portrait frames (H > W) and rotate them 90 degrees counter-clockwise along with camera intrinsics $K$, and dynamically select target working resolution matching the input image aspect ratio (16:9 -> 1280x720, 4:3 -> 1280x960), eliminating non-uniform stretching and squishing.
4. **Build & Test Verification**: Verified clean Xcode build with `xcodebuild` (**BUILD SUCCEEDED**) and ran end-to-end Python test suite (`python3 -m unittest discover -s tests`), passing all **26/26 tests cleanly**.
