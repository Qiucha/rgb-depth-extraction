# Issue #17: Task - End-to-End Live Streaming Integration Test Suite & Pipeline Verification

## Question

How can we create an automated Python integration test suite that simulates real-time WebSocket dual-stream binary frame transmission, verifies server packet decoding, checks non-blocking visual digest updating, and validates dataset manifest creation?

## Context & Requirements

To prevent regressions in the iPhone live capture bridge and ensure the depth extraction system processes incoming frames smoothly:
1. We need a unit/integration test suite (`tests/test_live_streaming_pipeline.py`) that boots an `IOSBridgeServer` on a test port.
2. The test simulates an iOS client encoding binary packets (`HEADER_MAGIC`, `HEADER_STRUCT`, metadata JSON, Main JPEG, Ultra-Wide JPEG).
3. The test streams multiple frames over WebSockets to the bridge server, verifying that:
   - Packets are correctly unpacked by `BinaryPacketDecoder`.
   - Frame sequences are saved to `data/live_captures/`.
   - `dataset_manifest.json` is generated upon disconnection.
   - Non-blocking `_process_and_generate_live_digest` runs without throwing errors or blocking the event loop.
4. Run python test suite and verify clean exit (code 0).

## Resolution

1. Created automated end-to-end integration test suite in `tests/test_live_streaming_pipeline.py`.
2. Verified `IOSBridgeServer` WebSocket handshake, binary frame decoding, sequence directory hierarchy creation, and `dataset_manifest.json` generation.
3. Executed full test suite (`python3 -m unittest discover tests`). All 24 tests passed cleanly (**Ran 24 tests in 5.124s, OK**).
