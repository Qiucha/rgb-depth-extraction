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

public class MultiCamSessionManager: NSObject, AVCaptureDataOutputSynchronizerDelegate {
    private let captureSession = AVCaptureMultiCamSession()
    private var dataSynchronizer: AVCaptureDataOutputSynchronizer?
    private var frameCounter: UInt64 = 0
    private var mainOutputRef: AVCaptureVideoDataOutput?
    private var uwOutputRef: AVCaptureVideoDataOutput?

    public var onFrameCaptured: ((StereoFramePacket) -> Void)?

    public override init() {
        super.init()
    }

    public func checkMultiCamSupport() -> Bool {
        return AVCaptureMultiCamSession.isMultiCamSupported
    }

    public func configureSession(targetWidth: Int32 = 1920, targetHeight: Int32 = 1080) throws {
        guard AVCaptureMultiCamSession.isMultiCamSupported else {
            throw NSError(domain: "MultiCamSession", code: 1, userInfo: [NSLocalizedDescriptionKey: "Hardware multi-cam not supported on this device."])
        }

        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        let pixelFormat = Int(kCVPixelFormatType_32BGRA)

        // 1. Configure Main Wide Camera
        guard let mainDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let mainInput = try? AVCaptureDeviceInput(device: mainDevice),
              captureSession.canAddInput(mainInput) else {
            throw NSError(domain: "MultiCamSession", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to add Main camera input."])
        }
        captureSession.addInputWithNoConnections(mainInput)

        try configureDeviceFormat(device: mainDevice, targetWidth: targetWidth, targetHeight: targetHeight)

        let mainOutput = AVCaptureVideoDataOutput()
        mainOutput.alwaysDiscardsLateVideoFrames = true
        mainOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]

        guard captureSession.canAddOutput(mainOutput) else {
            throw NSError(domain: "MultiCamSession", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to add Main camera output."])
        }
        captureSession.addOutputWithNoConnections(mainOutput)
        self.mainOutputRef = mainOutput

        guard let mainPort = mainInput.ports.first(where: { $0.mediaType == .video }) else {
            throw NSError(domain: "MultiCamSession", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to find Main camera video port."])
        }

        let mainConnection = AVCaptureConnection(inputPorts: [mainPort], output: mainOutput)
        if mainConnection.isCameraIntrinsicMatrixDeliverySupported {
            mainConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }
        captureSession.addConnection(mainConnection)

        // 2. Configure Ultra-Wide Camera
        guard let uwDevice = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back),
              let uwInput = try? AVCaptureDeviceInput(device: uwDevice),
              captureSession.canAddInput(uwInput) else {
            throw NSError(domain: "MultiCamSession", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to add Ultra-Wide camera input."])
        }
        captureSession.addInputWithNoConnections(uwInput)

        try configureDeviceFormat(device: uwDevice, targetWidth: targetWidth, targetHeight: targetHeight)

        let uwOutput = AVCaptureVideoDataOutput()
        uwOutput.alwaysDiscardsLateVideoFrames = true
        uwOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]

        guard captureSession.canAddOutput(uwOutput) else {
            throw NSError(domain: "MultiCamSession", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to add Ultra-Wide camera output."])
        }
        captureSession.addOutputWithNoConnections(uwOutput)
        self.uwOutputRef = uwOutput

        guard let uwPort = uwInput.ports.first(where: { $0.mediaType == .video }) else {
            throw NSError(domain: "MultiCamSession", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to find Ultra-Wide camera video port."])
        }

        let uwConnection = AVCaptureConnection(inputPorts: [uwPort], output: uwOutput)
        if uwConnection.isCameraIntrinsicMatrixDeliverySupported {
            uwConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }
        captureSession.addConnection(uwConnection)

        // 3. Hardware Data Output Synchronizer
        dataSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [mainOutput, uwOutput])
        dataSynchronizer?.setDelegate(self, queue: DispatchQueue(label: "com.robotics.multicam.sync", qos: .userInitiated))
    }

    private func configureDeviceFormat(device: AVCaptureDevice, targetWidth: Int32, targetHeight: Int32) throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        let targetFormat = device.formats.first { format in
            guard format.isMultiCamSupported else { return false }
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return dims.width == targetWidth && dims.height == targetHeight
        } ?? device.formats.first { $0.isMultiCamSupported }

        if let format = targetFormat {
            device.activeFormat = format
        }
    }

    public func startSession() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    public func stopSession() {
        captureSession.stopRunning()
    }

    // MARK: - AVCaptureDataOutputSynchronizerDelegate
    public func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        guard let mainOutput = mainOutputRef,
              let uwOutput = uwOutputRef,
              let mainData = synchronizedDataCollection.synchronizedData(for: mainOutput) as? AVCaptureSynchronizedSampleBufferData,
              let uwData = synchronizedDataCollection.synchronizedData(for: uwOutput) as? AVCaptureSynchronizedSampleBufferData else {
            return
        }

        let mainBuffer = mainData.sampleBuffer
        let uwBuffer = uwData.sampleBuffer
        let ptsNS = UInt64(CMTimeGetSeconds(mainData.timestamp) * 1e9)
        self.frameCounter += 1

        guard let mainJPEG = sampleBufferToJPEG(mainBuffer),
              let uwJPEG = sampleBufferToJPEG(uwBuffer) else { return }

        let mainIntrinsics = extractIntrinsics(from: mainBuffer)
        let mainMatrix = mainIntrinsics ?? matrix_float3x3(rows: [
            SIMD3<Float>(1400.0, 0, 960.0),
            SIMD3<Float>(0, 1400.0, 540.0),
            SIMD3<Float>(0, 0, 1.0)
        ])

        let uwIntrinsics = extractIntrinsics(from: uwBuffer)
        let uwMatrix = uwIntrinsics ?? matrix_float3x3(rows: [
            SIMD3<Float>(600.0, 0, 960.0),
            SIMD3<Float>(0, 600.0, 540.0),
            SIMD3<Float>(0, 0, 1.0)
        ])

        let metaDict: [String: Any] = [
            "frame_id": self.frameCounter,
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
            ]
        ]

        guard let metaJSON = try? JSONSerialization.data(withJSONObject: metaDict) else { return }

        let packet = StereoFramePacket(
            frameID: self.frameCounter,
            timestampNS: ptsNS,
            mainJPEG: mainJPEG,
            ultrawideJPEG: uwJPEG,
            metadataJSON: metaJSON
        )

        self.onFrameCaptured?(packet)
    }

    private func sampleBufferToJPEG(_ sampleBuffer: CMSampleBuffer) -> Data? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let context = CIContext(options: [.useSoftwareRenderer: false])
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent, format: .RGBA8, colorSpace: colorSpace) else { return nil }
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
