//
//  ContentView.swift
//  DualCamStereoCapture
//
//  SwiftUI User Interface for launching iPhone Dual-Camera streaming session to Python server.
//

import SwiftUI

struct ContentView: View {
    @State private var serverIP: String = "10.201.120.111"
    @State private var serverPort: String = "8765"
    @State private var isStreaming: Bool = false
    @State private var statusMessage: String = "Ready to connect"

    private let multiCamManager = MultiCamSessionManager()
    private let streamer = WebSocketStreamer()

    var body: some View {
        VStack(spacing: 24) {
            Text("iPhone Stereo Depth Capture")
                .font(.title2)
                .bold()

            VStack(alignment: .leading, spacing: 12) {
                Text("Python Bridge Server:")
                    .font(.caption)
                    .foregroundColor(.gray)

                HStack {
                    TextField("Server IP Address", text: $serverIP)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.none)
                        .disableAutocorrection(true)

                    TextField("Port", text: $serverPort)
                        .frame(width: 80)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .keyboardType(.numberPad)
                }
            }
            .padding(.horizontal)

            Text(statusMessage)
                .font(.subheadline)
                .foregroundColor(isStreaming ? .green : .secondary)

            Button(action: toggleStreaming) {
                HStack {
                    Image(systemName: isStreaming ? "stop.fill" : "record.circle")
                    Text(isStreaming ? "Stop Streaming" : "Start Live Streaming")
                }
                .font(.headline)
                .foregroundColor(.white)
                .padding()
                .frame(maxWidth: .infinity)
                .background(isStreaming ? Color.red : Color.blue)
                .cornerRadius(12)
            }
            .padding(.horizontal)

            Spacer()
        }
        .padding(.top, 40)
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

        if !multiCamManager.checkMultiCamSupport() {
            statusMessage = "MultiCam not supported on this iPhone model."
            return
        }

        do {
            try multiCamManager.configureSession()
            multiCamManager.onFrameCaptured = { packet in
                if isStreaming {
                    streamer.sendPacket(packet)
                }
            }
            statusMessage = "Camera configured (Dual Wide + Ultra-Wide)"
        } catch {
            statusMessage = "Config Error: \(error.localizedDescription)"
        }
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
            streamer.connect(url: url)
            multiCamManager.startSession()
            isStreaming = true
            statusMessage = "Live Streaming to ws://\(serverIP):\(serverPort)"
        }
    }
}
