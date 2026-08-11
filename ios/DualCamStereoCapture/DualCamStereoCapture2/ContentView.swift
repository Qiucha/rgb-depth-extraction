//
//  ContentView.swift
//  DualCamStereoCapture
//
//  SwiftUI User Interface for launching iPhone Dual-Camera snapshot & streaming depth extraction.
//

import SwiftUI

struct ContentView: View {
    @State private var serverIP: String = "10.201.120.111"
    @State private var serverPort: String = "8765"
    @State private var snapshotPort: String = "8766"
    @State private var isStreaming: Bool = false
    @State private var isProcessingSnapshot: Bool = false
    @State private var statusMessage: String = "Ready for capture"

    private let multiCamManager = MultiCamSessionManager()
    private let streamer = WebSocketStreamer()
    private let singleSnapshotManager = SingleSnapshotCaptureManager()

    var body: some View {
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
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // 1. main_image
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"main_image\"; filename=\"main.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(snapshot.mainJPEG)
        body.append("\r\n".data(using: .utf8)!)

        // 2. ultrawide_image
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"ultrawide_image\"; filename=\"ultrawide.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(snapshot.ultrawideJPEG)
        body.append("\r\n".data(using: .utf8)!)

        // 3. metadata
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"metadata\"\r\n\r\n".data(using: .utf8)!)
        body.append(snapshot.metadataJSON)
        body.append("\r\n".data(using: .utf8)!)

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isProcessingSnapshot = false
                if let error = error {
                    self.statusMessage = "Upload Error: \(error.localizedDescription)"
                    return
                }
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let status = json["status"] as? String,
                      status == "trusted" || status == "diagnostic" else {
                    self.statusMessage = "Server Processing Error"
                    return
                }
                let trusted = (json["trusted_depth_eligible"] as? Bool) == true
                self.statusMessage = trusted
                    ? "✅ Trusted Depth Ready! Dashboard: http://\(self.serverIP):8080"
                    : "⚠️ Diagnostic Depth Ready — review calibration evidence: http://\(self.serverIP):8080"
            }
        }.resume()
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
