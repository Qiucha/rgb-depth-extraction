# Issue #14: Task - Fix iOS WebSocket Handshake Race Condition & Streamer State Machine

## Question

How can we eliminate the WebSocket connection race condition in `WebSocketStreamer.swift` and `ContentView.swift` where initial camera frames are sent before the socket handshake completes, causing `task.send` errors and setting `isStreaming = false` permanently?

## Context & Root Cause

In `ContentView.swift`, when the user taps "Start Live Streaming", `streamer.connect(url)` calls `webSocketTask.resume()` and sets `isConnected = true` immediately. However, the WebSocket TCP/HTTP handshake is asynchronous and takes ~50-200ms. In the meantime, `multiCamManager.startSession()` begins delivering camera frames within ~30ms. Frame 1 triggers `streamer.sendPacket(packet)`, which invokes `task.send(message)`. Because the underlying socket is not yet open, `task.send` fails, triggering the error handler:
```swift
self?.isConnected = false
self?.onError?("Send Error: \(error.localizedDescription)")
```
`ContentView` receives `onError` and sets `isStreaming = false`, causing all subsequent camera frames to be dropped silently (`if self.isStreaming { streamer.sendPacket(packet) }`).

## Resolution

1. Replaced primitive boolean flag in `WebSocketStreamer.swift` with an explicit `WebSocketConnectionState` enum (`disconnected`, `connecting`, `connected`, `failed`).
2. Implemented `URLSessionWebSocketDelegate` in `WebSocketStreamer`:
   - `urlSession(_:webSocketTask:didOpenWithProtocol:)` updates state to `.connected` and triggers `onConnect`.
   - `urlSession(_:task:didCompleteWithError:)` handles connection errors cleanly.
3. Updated `sendPacket(_:)` to inspect `state == .connected`. If a frame arrives while `state == .connecting`, `sendPacket` silently drops frame 1 without throwing an error or changing `isStreaming` state.
4. Updated `ContentView.swift` to handle `onConnect` callback and maintain streaming state across socket initialization.
5. Verified clean build with `xcodebuild` (**BUILD SUCCEEDED**).
