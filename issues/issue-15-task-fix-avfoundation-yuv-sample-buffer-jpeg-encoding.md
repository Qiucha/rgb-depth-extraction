# Issue #15: Task - Fix AVFoundation YUV Sample Buffer JPEG Encoding & Frame Discarding

## Question

Why are dual-camera sample buffers silently discarded in `MultiCamSessionManager.swift`, and how do we ensure zero-copy / high-performance JPEG encoding for both Main and Ultra-Wide frames?

## Context & Root Cause

`MultiCamSessionManager.swift` sets output video settings to `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`:
```swift
let pixelFormat = Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
```
When `dataOutputSynchronizer` receives sample buffers, it calls `sampleBufferToJPEG`:
```swift
private func sampleBufferToJPEG(_ sampleBuffer: CMSampleBuffer) -> Data? {
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
    let ciImage = CIImage(cvImageBuffer: imageBuffer)
    guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
    let uiImage = UIImage(cgImage: cgImage)
    return uiImage.jpegData(compressionQuality: 0.7)
}
```
For bi-planar YUV pixel buffers (`420YpCbCr8BiPlanarFullRange`), `ciContext.createCGImage` returns `nil` unless explicit color space matrix attachments are provided. When `createCGImage` returns `nil`, `sampleBufferToJPEG` returns `nil`, hitting line 260:
```swift
guard let mainJPEG = sampleBufferToJPEG(mainBuffer),
      let uwJPEG = sampleBufferToJPEG(uwBuffer) else { return }
```
This silently returns and discards 100% of captured camera frames!

## Resolution

1. Updated video output pixel format in `MultiCamSessionManager.swift` to `kCVPixelFormatType_32BGRA` for direct CoreImage / CGImage compatibility.
2. Updated `sampleBufferToJPEG` with explicit `CGColorSpaceCreateDeviceRGB()` and `.BGRA8` rendering format, plus fallback to `ciContext.createCGImage`.
3. Added diagnostic warning logs in `dataOutputSynchronizer` if frame conversion ever fails.
4. Verified clean build with `xcodebuild` (**BUILD SUCCEEDED**).
