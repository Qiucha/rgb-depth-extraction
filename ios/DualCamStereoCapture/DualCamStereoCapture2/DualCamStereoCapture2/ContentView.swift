//
//  ContentView.swift
//  DualCamStereoCapture
//
//  SwiftUI User Interface for launching iPhone Dual-Camera snapshot & streaming depth extraction.
//

import SwiftUI
import UIKit

private struct SnapshotResultPreview: View {
    let mainImage: UIImage
    let depthOverlayURL: URL?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Latest Stereo Processing Result")
                .font(.headline)

            HStack(alignment: .top, spacing: 12) {
                previewCard(title: "Captured Main") {
                    Image(uiImage: mainImage)
                        .resizable()
                        .scaledToFit()
                }

                previewCard(title: "Depth Overlay") {
                    if let depthOverlayURL {
                        AsyncImage(url: depthOverlayURL) { phase in
                            switch phase {
                            case .success(let image):
                                image.resizable().scaledToFit()
                            case .failure:
                                ContentUnavailableView(
                                    "Overlay unavailable",
                                    systemImage: "exclamationmark.triangle"
                                )
                            default:
                                ProgressView()
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                            }
                        }
                    } else {
                        ProgressView("Processing depth…")
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
            }
            .frame(height: 180)
        }
        .padding(12)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func previewCard<Content: View>(
        title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(spacing: 6) {
            content()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct ContentView: View {
    @Environment(\.openURL) private var openURL
    @AppStorage("stereoServerIP") private var serverIP: String = "192.168.8.192"
    @State private var serverPort: String = "8765"
    @State private var snapshotPort: String = "8766"
    @State private var isStreaming: Bool = false
    @State private var isProcessingSnapshot: Bool = false
    @State private var statusMessage: String = "Ready for capture"
    @State private var capturedMainImage: UIImage?
    @State private var depthOverlayURL: URL?

    private let multiCamManager = MultiCamSessionManager()
    private let streamer = WebSocketStreamer()
    private let singleSnapshotManager = SingleSnapshotCaptureManager()
    private let snapshotUploadClient = SnapshotUploadClient()

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("iPhone Stereo Depth Capture")
                    .font(.title2)
                    .bold()

            VStack(alignment: .leading, spacing: 12) {
                Text("Python Server Host IP:")
                    .font(.caption)
                    .foregroundColor(.gray)

                HStack {
                    TextField("Server IP Address", text: $serverIP)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                }
            }
            .padding(.horizontal)

            Text(statusMessage)
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundColor(isStreaming ? .green : (isProcessingSnapshot ? .orange : .secondary))
                .padding(.horizontal)

            // Section 1: Single Snapshot Photo Pair Capture (Recommended)
            VStack(spacing: 12) {
                Button(action: captureAndUploadSnapshot) {
                    HStack {
                        if isProcessingSnapshot {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .padding(.trailing, 4)
                        } else {
                            Image(systemName: "camera.fill")
                        }
                        Text(isProcessingSnapshot ? "Capturing & Processing..." : "Capture Photo Pair & Extract Depth")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(isProcessingSnapshot ? Color.gray : Color.purple)
                    .cornerRadius(12)
                }
                .disabled(isProcessingSnapshot || isStreaming)

                Text("Single-shot mode: Captures synchronized Main + Ultra-Wide photos & POSTs to server (port 8766)")
                    .font(.caption2)
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)

                Button(action: openDigest) {
                    Label("Open Latest Digest", systemImage: "chart.xyaxis.line")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                if let capturedMainImage {
                    SnapshotResultPreview(
                        mainImage: capturedMainImage,
                        depthOverlayURL: depthOverlayURL
                    )
                }
            }
            .padding(.horizontal)

            Divider()
                .padding(.vertical, 8)

            // Section 2: Real-Time Stream Session
            VStack(spacing: 12) {
                Button(action: toggleStreaming) {
                    HStack {
                        Image(systemName: isStreaming ? "stop.fill" : "record.circle")
                        Text(isStreaming ? "Stop Live Streaming" : "Start Live Video Stream")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(isStreaming ? Color.red : Color.blue)
                    .cornerRadius(12)
                }
                .disabled(isProcessingSnapshot)

                Text("Stream mode: Continuous WebSockets streaming on port 8765")
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            .padding(.horizontal)

                Spacer()
            }
            .padding(.top, 30)
        }
        .onAppear {
            setupCapture()
        }
    }

    private func setupCapture() {
        streamer.onError = { errorText in
            DispatchQueue.main.async {
                self.isStreaming = false
                self.statusMessage = "⚠️ \(errorText)"
            }
        }
        streamer.onConnect = {
            DispatchQueue.main.async {
                self.statusMessage = "🟢 Live Streaming Active (Connected)"
            }
        }

        if !multiCamManager.checkMultiCamSupport() {
            statusMessage = "MultiCam not supported on this iPhone model or when running in Simulator."
            return
        }

        multiCamManager.checkCameraAuthorization { result in
            switch result {
            case .success:
                statusMessage = "Camera Authorized & Ready"
            case .failure(let error):
                statusMessage = "Camera Error: \(error.localizedDescription)"
            }
        }
    }

    private func captureAndUploadSnapshot() {
        guard let serverURL = URL(string: "http://\(serverIP):\(snapshotPort)/api/upload_snapshot") else {
            statusMessage = "Invalid Server URL"
            return
        }

        isProcessingSnapshot = true
        statusMessage = "Capturing dual photo snapshot..."

        singleSnapshotManager.captureSnapshot { result in
            switch result {
            case .success(let snapshot):
                DispatchQueue.main.async {
                    self.capturedMainImage = UIImage(data: snapshot.mainJPEG)
                    self.depthOverlayURL = nil
                    self.statusMessage = "Uploading snapshot to server..."
                }
                self.uploadSnapshot(snapshot: snapshot, to: serverURL)
            case .failure(let error):
                DispatchQueue.main.async {
                    self.isProcessingSnapshot = false
                    self.statusMessage = "Capture Error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func uploadSnapshot(snapshot: StereoPhotoSnapshot, to url: URL) {
        snapshotUploadClient.upload(snapshot, to: url) { result in
            DispatchQueue.main.async {
                self.isProcessingSnapshot = false
                switch result {
                case .failure(let error):
                    self.statusMessage = "Upload Error: \(error.localizedDescription)"
                case .success(let preview):
                    self.statusMessage = preview.trustedDepthEligible
                        ? "✅ Trusted Depth Ready"
                        : "⚠️ Diagnostic Depth Ready — review calibration evidence"
                    guard let artifactURL = URL(
                        string: preview.depthOverlayPath, relativeTo: url
                    ),
                    var components = URLComponents(
                        url: artifactURL.absoluteURL,
                        resolvingAgainstBaseURL: true
                   ) else {
                        self.statusMessage = "Server Processing Error"
                        return
                    }
                    components.queryItems = [
                        URLQueryItem(
                            name: "capture",
                            value: String(snapshot.timestampNS)
                        )
                    ]
                    self.depthOverlayURL = components.url
                }
            }
        }
    }

    private func openDigest() {
        guard let url = URL(string: "http://\(serverIP):\(snapshotPort)/") else {
            statusMessage = "Invalid Server URL"
            return
        }
        openURL(url)
    }

    private func toggleStreaming() {
        if isStreaming {
            streamer.disconnect()
            multiCamManager.stopSession()
            isStreaming = false
            statusMessage = "Streaming stopped."
        } else {
            guard let url = URL(string: "ws://\(serverIP):\(serverPort)") else {
                statusMessage = "Invalid Server URL"
                return
            }

            do {
                try multiCamManager.configureSession()
                multiCamManager.onFrameCaptured = { packet in
                    if self.isStreaming {
                        self.streamer.sendPacket(packet)
                    }
                }
            } catch {
                statusMessage = "Config Error: \(error.localizedDescription)"
            }

            streamer.connect(url: url)
            multiCamManager.startSession()
            isStreaming = true
            statusMessage = "Connecting to ws://\(serverIP):\(serverPort)..."
        }
    }
}
