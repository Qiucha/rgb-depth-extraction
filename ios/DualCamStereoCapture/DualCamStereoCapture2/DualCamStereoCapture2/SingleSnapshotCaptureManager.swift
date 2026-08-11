//
//  SingleSnapshotCaptureManager.swift
//  DualCamStereoCapture
//
//  Manager for capturing a single synchronized dual-camera photo pair (Main Wide + Ultra-Wide) on demand.
//

import Foundation
import AVFoundation
import CoreImage
import UIKit
import CoreMedia
import simd

public struct StereoPhotoSnapshot {
    public let timestampNS: UInt64
    public let mainJPEG: Data
    public let ultrawideJPEG: Data
    public let metadataJSON: Data
}

public class SingleSnapshotCaptureManager: NSObject {
    private let multiCamManager = MultiCamSessionManager()
    private var pendingCompletion: ((Result<StereoPhotoSnapshot, Error>) -> Void)?
    private var isCapturing = false

    public override init() {
        super.init()
    }

    public func captureSnapshot(completion: @escaping (Result<StereoPhotoSnapshot, Error>) -> Void) {
        guard !isCapturing else {
            completion(.failure(NSError(domain: "SingleSnapshot", code: 1, userInfo: [NSLocalizedDescriptionKey: "Capture already in progress"])))
            return
        }

        isCapturing = true
        pendingCompletion = completion

        multiCamManager.checkCameraAuthorization { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success:
                do {
                    try self.multiCamManager.configureSession()
                    var warmupFramesLeft = 5
                    self.multiCamManager.onFrameCaptured = { [weak self] packet in
                        guard let self = self, self.isCapturing else { return }
                        if warmupFramesLeft > 0 {
                            warmupFramesLeft -= 1
                            return
                        }
                        print("[SingleSnapshotCaptureManager] Synchronized dual-camera snapshot frame captured successfully!")
                        self.isCapturing = false
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                            self.multiCamManager.stopSession()
                        }

                        let snapshot = StereoPhotoSnapshot(
                            timestampNS: packet.timestampNS,
                            mainJPEG: packet.mainJPEG,
                            ultrawideJPEG: packet.ultrawideJPEG,
                            metadataJSON: packet.metadataJSON
                        )
                        let cb = self.pendingCompletion
                        self.pendingCompletion = nil
                        DispatchQueue.main.async {
                            cb?(.success(snapshot))
                        }
                    }
                    self.multiCamManager.startSession()
                } catch {
                    self.isCapturing = false
                    let cb = self.pendingCompletion
                    self.pendingCompletion = nil
                    DispatchQueue.main.async {
                        cb?(.failure(error))
                    }
                }
            case .failure(let error):
                self.isCapturing = false
                let cb = self.pendingCompletion
                self.pendingCompletion = nil
                DispatchQueue.main.async {
                    cb?(.failure(error))
                }
            }
        }
    }
}
