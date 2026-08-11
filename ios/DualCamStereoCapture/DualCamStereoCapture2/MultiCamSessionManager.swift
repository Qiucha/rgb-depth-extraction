//
//  MultiCamSessionManager.swift
//  DualCamStereoCapture
//
//  AVFoundation AVCaptureMultiCamSession manager streaming synchronized Main + Ultra-Wide frames with dynamic intrinsics.
//

import Foundation
import AVFoundation
import CoreImage
import UIKit
import CoreMedia
import simd

public struct StereoFramePacket {
    public let frameID: UInt64
    public let timestampNS: UInt64
    public let mainJPEG: Data
    public let ultrawideJPEG: Data
    public let metadataJSON: Data
}

public enum MultiCamSessionError: LocalizedError {
    case multiCamNotSupported
    case cameraPermissionDenied
    case mainInputCreationFailed
    case mainOutputCreationFailed
    case mainVideoPortNotFound
    case ultrawideInputCreationFailed
    case ultrawideOutputCreationFailed
    case ultrawideVideoPortNotFound
    case hardwareCostBudgetExceeded(cost: Float)

    public var errorDescription: String? {
        switch self {
        case .multiCamNotSupported:
            return "Hardware multi-cam not supported on this device or when running on Xcode Simulator."
        case .cameraPermissionDenied:
            return "Camera access is denied or restricted in iOS Privacy Settings."
        case .mainInputCreationFailed:
            return "Failed to add Main camera input to capture session."
        case .mainOutputCreationFailed:
            return "Failed to add Main camera output to capture session."
        case .mainVideoPortNotFound:
            return "Failed to find Main camera video port."
        case .ultrawideInputCreationFailed:
            return "Failed to add Ultra-Wide camera input to capture session."
        case .ultrawideOutputCreationFailed:
            return "Failed to add Ultra-Wide camera output to capture session."
        case .ultrawideVideoPortNotFound:
            return "Failed to find Ultra-Wide camera video port."
        case .hardwareCostBudgetExceeded(let cost):
            return "Exceeded AVCaptureMultiCamSession hardware cost budget (hardwareCost: \(cost) > 1.0). Lower active device resolutions or framerates."
        }
    }
}

public class MultiCamSessionManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let captureSession = AVCaptureMultiCamSession()
    private var frameCounter: UInt64 = 0
    private var mainOutputRef: AVCaptureVideoDataOutput?
    private var uwOutputRef: AVCaptureVideoDataOutput?
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let processingQueue = DispatchQueue(label: "com.robotics.multicam.processing", qos: .userInitiated)
    private let syncQueueMain = DispatchQueue(label: "com.robotics.multicam.main", qos: .userInitiated)
    private let syncQueueUW = DispatchQueue(label: "com.robotics.multicam.uw", qos: .userInitiated)

    private var latestMainSampleBuffer: CMSampleBuffer?
    private var latestUWSampleBuffer: CMSampleBuffer?
    private let bufferLock = NSLock()

    // Cached hardware bandwidth cost to prevent session lock contention on frame delivery thread
    private var cachedHardwareBandwidthCost: Float = 0.0

    public var onFrameCaptured: ((StereoFramePacket) -> Void)?

    public override init() {
        super.init()
    }

    public func checkMultiCamSupport() -> Bool {
        return AVCaptureMultiCamSession.isMultiCamSupported
    }

    public func checkCameraAuthorization(completion: @escaping (Result<Void, MultiCamSessionError>) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            DispatchQueue.main.async { completion(.success(())) }
        case .denied, .restricted:
            DispatchQueue.main.async { completion(.failure(.cameraPermissionDenied)) }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        completion(.success(()))
                    } else {
                        completion(.failure(.cameraPermissionDenied))
                    }
                }
            }
        @unknown default:
            DispatchQueue.main.async { completion(.failure(.cameraPermissionDenied)) }
        }
    }

    public func configureSession(targetWidth: Int32 = 1920, targetHeight: Int32 = 1080) throws {
        guard AVCaptureMultiCamSession.isMultiCamSupported else {
            throw MultiCamSessionError.multiCamNotSupported
        }

        let status = AVCaptureDevice.authorizationStatus(for: .video)
        guard status == .authorized else {
            throw MultiCamSessionError.cameraPermissionDenied
        }

        struct Resolution {
            let width: Int32
            let height: Int32
        }

        let candidateResolutions: [Resolution] = [
            Resolution(width: targetWidth, height: targetHeight),
            Resolution(width: 1280, height: 720),
            Resolution(width: 640, height: 480)
        ]

        var configuredSuccessfully = false
        var lastCost: Float = 0.0

        for res in candidateResolutions {
            captureSession.beginConfiguration()

            for input in captureSession.inputs { captureSession.removeInput(input) }
            for output in captureSession.outputs { captureSession.removeOutput(output) }

            // Use 32BGRA pixel format for direct CoreImage / CGImage JPEG rendering compatibility
            let pixelFormat = Int(kCVPixelFormatType_32BGRA)

            // 1. Configure Main Wide Camera
            guard let mainDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let mainInput = try? AVCaptureDeviceInput(device: mainDevice),
                  captureSession.canAddInput(mainInput) else {
                captureSession.commitConfiguration()
                throw MultiCamSessionError.mainInputCreationFailed
            }
            captureSession.addInputWithNoConnections(mainInput)

            try configureDeviceFormat(device: mainDevice, targetWidth: res.width, targetHeight: res.height)

            let mainOutput = AVCaptureVideoDataOutput()
            mainOutput.alwaysDiscardsLateVideoFrames = true
            mainOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]
            mainOutput.setSampleBufferDelegate(self, queue: syncQueueMain)

            guard captureSession.canAddOutput(mainOutput) else {
                captureSession.commitConfiguration()
                throw MultiCamSessionError.mainOutputCreationFailed
            }
            captureSession.addOutputWithNoConnections(mainOutput)
            self.mainOutputRef = mainOutput

            guard let mainPort = mainInput.ports.first(where: { $0.mediaType == .video }) else {
                captureSession.commitConfiguration()
                throw MultiCamSessionError.mainVideoPortNotFound
            }

            let mainConnection = AVCaptureConnection(inputPorts: [mainPort], output: mainOutput)
            if mainConnection.isVideoOrientationSupported {
                mainConnection.videoOrientation = .landscapeRight
            }
            if mainConnection.isCameraIntrinsicMatrixDeliverySupported {
                mainConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
            captureSession.addConnection(mainConnection)

            // 2. Configure Ultra-Wide Camera
            guard let uwDevice = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back),
                  let uwInput = try? AVCaptureDeviceInput(device: uwDevice),
                  captureSession.canAddInput(uwInput) else {
                captureSession.commitConfiguration()
                throw MultiCamSessionError.ultrawideInputCreationFailed
            }
            captureSession.addInputWithNoConnections(uwInput)

            try configureDeviceFormat(device: uwDevice, targetWidth: res.width, targetHeight: res.height)

            let uwOutput = AVCaptureVideoDataOutput()
            uwOutput.alwaysDiscardsLateVideoFrames = true
            uwOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]
            uwOutput.setSampleBufferDelegate(self, queue: syncQueueUW)

            guard captureSession.canAddOutput(uwOutput) else {
                captureSession.commitConfiguration()
                throw MultiCamSessionError.ultrawideOutputCreationFailed
            }
            captureSession.addOutputWithNoConnections(uwOutput)
            self.uwOutputRef = uwOutput

            guard let uwPort = uwInput.ports.first(where: { $0.mediaType == .video }) else {
                captureSession.commitConfiguration()
                throw MultiCamSessionError.ultrawideVideoPortNotFound
            }

            let uwConnection = AVCaptureConnection(inputPorts: [uwPort], output: uwOutput)
            if uwConnection.isVideoOrientationSupported {
                uwConnection.videoOrientation = .landscapeRight
            }
            if uwConnection.isCameraIntrinsicMatrixDeliverySupported {
                uwConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
            captureSession.addConnection(uwConnection)

            lastCost = captureSession.hardwareCost
            captureSession.commitConfiguration()

            if captureSession.hardwareCost <= 1.0 {
                configuredSuccessfully = true
                self.cachedHardwareBandwidthCost = captureSession.hardwareCost
                print("[MultiCamSessionManager] Session configured successfully at \(res.width)x\(res.height) with hardwareCost: \(captureSession.hardwareCost)")
                break
            } else {
                print("[MultiCamSessionManager] Hardware cost \(captureSession.hardwareCost) > 1.0 for resolution \(res.width)x\(res.height). Retrying lower format...")
            }
        }

        if !configuredSuccessfully {
            throw MultiCamSessionError.hardwareCostBudgetExceeded(cost: lastCost)
        }
    }

    private func configureDeviceFormat(device: AVCaptureDevice, targetWidth: Int32, targetHeight: Int32) throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        let targetFormat = device.formats.first { format in
            guard format.isMultiCamSupported else { return false }
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return (dims.width == targetWidth && dims.height == targetHeight) || (dims.width == targetHeight && dims.height == targetWidth)
        } ?? device.formats.first { $0.isMultiCamSupported }

        if let format = targetFormat {
            device.activeFormat = format
            // Lock frame rate to 30 FPS (supported by all iOS multi-cam formats, hardwareCost <= 0.60)
            let targetFrameDuration = CMTime(value: 1, timescale: 30)
            for range in format.videoSupportedFrameRateRanges {
                if range.minFrameDuration <= targetFrameDuration && targetFrameDuration <= range.maxFrameDuration {
                    device.activeVideoMinFrameDuration = targetFrameDuration
                    device.activeVideoMaxFrameDuration = targetFrameDuration
            // Lock focus mode and set fixed lens position to prevent focus hunt blur (1.0 = farthest / infinity focus)
            if device.isFocusModeSupported(.locked) {
                device.setFocusModeLocked(lensPosition: 1.0) { _ in }
                print("[MultiCamSessionManager] Locked focus mode to fixed lens position (1.0 - infinity focus)")
            } else if device.isFocusModeSupported(.autoFocus) {
                device.focusMode = .autoFocus
            }

            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
        }
    }

    public func startSession() {
        bufferLock.lock()
        latestMainSampleBuffer = nil
        latestUWSampleBuffer = nil
        bufferLock.unlock()

        if !captureSession.isRunning {
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        }
    }

    public func stopSession() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        bufferLock.lock()

        if output == mainOutputRef {
            latestMainSampleBuffer = sampleBuffer
        } else if output == uwOutputRef {
            latestUWSampleBuffer = sampleBuffer
        }

        guard let mainBuffer = latestMainSampleBuffer,
              let uwBuffer = latestUWSampleBuffer else {
            bufferLock.unlock()
            return
        }

        let mainPTS = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(mainBuffer))
        let uwPTS = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(uwBuffer))
        let timeDiff = abs(mainPTS - uwPTS)

        if timeDiff < 0.05 {
            // Matched synchronized pair within 50ms!
            latestMainSampleBuffer = nil
            latestUWSampleBuffer = nil
            bufferLock.unlock()

            self.frameCounter += 1
            let currentFrameID = self.frameCounter
            let ptsNS = UInt64(mainPTS * 1e9)

            guard let mainPixelBuffer = CMSampleBufferGetImageBuffer(mainBuffer),
                  let uwPixelBuffer = CMSampleBufferGetImageBuffer(uwBuffer) else {
                return
            }

            let mainIntrinsics = extractIntrinsics(from: mainBuffer)
            let uwIntrinsics = extractIntrinsics(from: uwBuffer)

            // Offload heavy JPEG encoding off sample delegate thread
            processingQueue.async { [weak self] in
                guard let self = self else { return }

                guard let mainJPEG = self.pixelBufferToJPEG(mainPixelBuffer),
                      let uwJPEG = self.pixelBufferToJPEG(uwPixelBuffer) else {
                    print("[MultiCamSessionManager] Warning: Failed to encode pixel buffer to JPEG payload for frame #\(currentFrameID)")
                    return
                }

                let mainMatrix = mainIntrinsics ?? matrix_float3x3(rows: [
                    SIMD3<Float>(1400.0, 0, 960.0),
                    SIMD3<Float>(0, 1400.0, 540.0),
                    SIMD3<Float>(0, 0, 1.0)
                ])

                let uwMatrix = uwIntrinsics ?? matrix_float3x3(rows: [
                    SIMD3<Float>(600.0, 0, 960.0),
                    SIMD3<Float>(0, 600.0, 540.0),
                    SIMD3<Float>(0, 0, 1.0)
                ])

                let metaDict: [String: Any] = [
                    "frame_id": currentFrameID,
                    "timestamp_pts_ns": ptsNS,
                    "main": [
                        "fx": mainMatrix[0][0], "fy": mainMatrix[1][1],
                        "cx": mainMatrix[0][2], "cy": mainMatrix[1][2]
                    ],
                    "ultrawide": [
                        "fx": uwMatrix[0][0], "fy": uwMatrix[1][1],
                        "cx": uwMatrix[0][2], "cy": uwMatrix[1][2]
                    ],
                    "extrinsics_uw_to_main": [
                        "rotation_matrix_3x3": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                        "translation_vector_mm": [19.5, 0.0, 0.0]
                    ],
                    "telemetry": [
                        "hardware_bandwidth_cost": self.cachedHardwareBandwidthCost,
                        "is_multi_cam_supported": AVCaptureMultiCamSession.isMultiCamSupported
                    ]
                ]

                guard let metaJSON = try? JSONSerialization.data(withJSONObject: metaDict) else { return }

                let packet = StereoFramePacket(
                    frameID: currentFrameID,
                    timestampNS: ptsNS,
                    mainJPEG: mainJPEG,
                    ultrawideJPEG: uwJPEG,
                    metadataJSON: metaJSON
                )

                self.onFrameCaptured?(packet)
            }
        } else {
            // Drop older buffer if timestamps are too far apart
            if mainPTS < uwPTS {
                latestMainSampleBuffer = nil
            } else {
                latestUWSampleBuffer = nil
            }
            bufferLock.unlock()
        }
    }

    private func pixelBufferToJPEG(_ imageBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent, format: .BGRA8, colorSpace: colorSpace) ?? ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            print("[MultiCamSessionManager] Error: Failed to render CGImage from pixel buffer")
            return nil
        }
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: 0.8)
    }

    private func extractIntrinsics(from sampleBuffer: CMSampleBuffer) -> matrix_float3x3? {
        guard let attachment = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) else {
            return nil
        }
        let matrixData = attachment as! CFData
        guard CFDataGetLength(matrixData) == MemoryLayout<matrix_float3x3>.size else { return nil }
        var matrix = matrix_float3x3()
        CFDataGetBytes(matrixData, CFRangeMake(0, MemoryLayout<matrix_float3x3>.size), withUnsafeMutablePointer(to: &matrix, { UnsafeMutableRawPointer($0).assumingMemoryBound(to: UInt8.self) }))
        return matrix
    }
}
