# Teaching Notes & User Preferences

- User requested code snippet explanations core to disparity extraction in `src/stereo_depth/sliding_window.py` and `wls_filter.py`.
- User asked: "Why don't we just use the method we used to extract GT?"
- Key teaching point: Ground truth is captured via Active Structured Light (projecting coded light patterns), which requires controlled indoor conditions, static scenes, and expensive projector hardware. Passive stereo relies solely on ambient light and dual cameras, enabling real-time outdoor robotics.
- **iOS MultiCam Live Streaming Diagnosis Document**: [ios_dualcam_streaming_diagnosis.md](file:///Users/q/Documents/Projects/robotics_neo/docs/spec/ios_dualcam_streaming_diagnosis.md) detailing the 4 root causes, 30 FPS duration lock, BGRA color conversion, and WebSocket state machine fixes.
