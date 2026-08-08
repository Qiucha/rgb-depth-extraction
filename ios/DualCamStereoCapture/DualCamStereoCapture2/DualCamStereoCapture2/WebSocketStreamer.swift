//
//  WebSocketStreamer.swift
//  DualCamStereoCapture
//
//  Binary packet encoder and URLSessionWebSocketTask network client for streaming to Python bridge.
//

import Foundation

public class WebSocketStreamer {
    private var webSocketTask: URLSessionWebSocketTask?
    private var isConnected = false

    public init() {}

    public func connect(url: URL) {
        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()
        isConnected = true
        print("[WebSocketStreamer] Connected to \(url)")
    }

    public func disconnect() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        isConnected = false
        print("[WebSocketStreamer] Disconnected")
    }

    public func sendPacket(_ packet: StereoFramePacket) {
        guard isConnected, let task = webSocketTask else { return }

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
        task.send(message) { error in
            if let error = error {
                print("[WebSocketStreamer] Error sending binary packet: \(error)")
            }
        }
    }
}
