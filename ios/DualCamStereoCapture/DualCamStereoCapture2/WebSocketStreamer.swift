//
//  WebSocketStreamer.swift
//  DualCamStereoCapture
//
//  Binary packet encoder and URLSessionWebSocketTask network client for streaming to Python bridge.
//

import Foundation

public enum WebSocketConnectionState {
    case disconnected
    case connecting
    case connected
    case failed
}

public class WebSocketStreamer: NSObject, URLSessionWebSocketDelegate {
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    public private(set) var state: WebSocketConnectionState = .disconnected
    private var isSending = false

    public var onError: ((String) -> Void)?
    public var onConnect: (() -> Void)?
    public var onDisconnect: (() -> Void)?

    public override init() {
        super.init()
    }

    public func connect(url: URL) {
        disconnect()

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10.0
        config.waitsForConnectivity = true
        
        let session = URLSession(configuration: config, delegate: self, delegateQueue: OperationQueue.main)
        self.urlSession = session

        let task = session.webSocketTask(with: url)
        task.maximumMessageSize = 32 * 1024 * 1024
        webSocketTask = task
        state = .connecting
        isSending = false
        print("[WebSocketStreamer] Connecting to \(url)...")
        task.resume()
    }

    public func disconnect() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        let wasConnected = (state == .connected)
        state = .disconnected
        isSending = false
        if wasConnected {
            onDisconnect?()
        }
        print("[WebSocketStreamer] Disconnected")
    }

    // MARK: - URLSessionWebSocketDelegate
    public func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        print("[WebSocketStreamer] WebSocket connected successfully (protocol: \(String(describing: `protocol`)))")
        state = .connected
        listenForMessages()
        DispatchQueue.main.async {
            self.onConnect?()
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("[WebSocketStreamer] Session completed with error: \(error.localizedDescription)")
            let wasConnectingOrConnected = (state == .connecting || state == .connected)
            state = .failed
            if wasConnectingOrConnected {
                DispatchQueue.main.async {
                    self.onError?("Connection Error: \(error.localizedDescription)")
                }
            }
        }
    }

    private func listenForMessages() {
        webSocketTask?.receive { [weak self] result in
            guard let self = self, self.state == .connected else { return }
            switch result {
            case .failure(let error):
                print("[WebSocketStreamer] WebSocket Receive Error: \(error.localizedDescription)")
                if self.state == .connected {
                    self.state = .failed
                    DispatchQueue.main.async {
                        self.onError?("Network Error: \(error.localizedDescription)")
                    }
                }
            case .success(let message):
                print("[WebSocketStreamer] Received server message: \(message)")
                self.listenForMessages()
            }
        }
    }

    private var lastSendTime = Date()

    public func sendPacket(_ packet: StereoFramePacket) {
        guard state == .connected, let task = webSocketTask else {
            // If still connecting or disconnected, gracefully ignore packet without error
            return
        }

        if isSending {
            if Date().timeIntervalSince(lastSendTime) > 0.5 {
                isSending = false
            } else {
                return // Skip frame if previous send is in-flight to prevent buffer congestion
            }
        }

        isSending = true
        lastSendTime = Date()

        // Construct 36-byte Header (ROBO, version=1, flags=0, frameID, pts, metaLen, mainLen, uwLen)
        var data = Data()

        // Magic: "ROBO"
        data.append(contentsOf: [0x52, 0x4F, 0x42, 0x4F]) // 'R', 'O', 'B', 'O'
        
        // Version (uint16 = 1)
        var version: UInt16 = 1
        data.append(Data(bytes: &version, count: 2))

        // Flags (uint16 = 0)
        var flags: UInt16 = 0
        data.append(Data(bytes: &flags, count: 2))

        // FrameID (uint64)
        var frameID = packet.frameID
        data.append(Data(bytes: &frameID, count: 8))

        // PTS (uint64)
        var pts = packet.timestampNS
        data.append(Data(bytes: &pts, count: 8))

        // Metadata JSON Length (uint32)
        var metaLen = UInt32(packet.metadataJSON.count)
        data.append(Data(bytes: &metaLen, count: 4))

        // Main JPEG Length (uint32)
        var mainLen = UInt32(packet.mainJPEG.count)
        data.append(Data(bytes: &mainLen, count: 4))

        // Ultra-Wide JPEG Length (uint32)
        var uwLen = UInt32(packet.ultrawideJPEG.count)
        data.append(Data(bytes: &uwLen, count: 4))

        // Append payloads
        data.append(packet.metadataJSON)
        data.append(packet.mainJPEG)
        data.append(packet.ultrawideJPEG)

        let message = URLSessionWebSocketTask.Message.data(data)
        task.send(message) { [weak self] error in
            self?.isSending = false
            if let error = error {
                print("[WebSocketStreamer] Error sending binary packet: \(error)")
                self?.state = .failed
                DispatchQueue.main.async {
                    self?.onError?("Send Error: \(error.localizedDescription)")
                }
            }
        }
    }
}
