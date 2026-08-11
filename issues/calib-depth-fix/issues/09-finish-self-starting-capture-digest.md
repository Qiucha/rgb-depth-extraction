Type: task
Status: resolved
Blocked by: 08

## Question

Finish the capture-to-digest pipeline so the user does not need to invoke the Python server manually, and ensure every completed digest displays the actual Main and Ultra-Wide source images alongside the derived rectification, disparity, confidence, and depth artifacts.

## Execution scope

The user's explicit request carries implementation through this map. Resolve the launch lifecycle at the iOS/Python seam, preserve the uploaded source images as digest artifacts, expose them through the digest interface, and verify the complete HTTP upload path with regression tests.

## Answer

The capture-to-digest experience is now a one-address, persistent flow:

- Installed `local.roboticsneo.snapshot-server` as a per-user macOS LaunchAgent with `RunAtLoad` and `KeepAlive`, using the repository's Homebrew Python/OpenCV environment. It starts at login, survives terminal closure, and restarts on port 8766.
- Collapsed capture upload, health, digest HTML, and digest artifacts onto `http://<Mac-IP>:8766/`. Port 8080 is no longer required.
- Added `/api/health`, retained `/api/upload_snapshot`, and made `/` serve either the latest Digest Dashboard or a capture-ready landing page.
- The iOS app remembers the Mac IP, uploads on port 8766, automatically opens the Digest Dashboard when processing finishes, and exposes an `Open Latest Digest` button.
- Every completed digest decodes and stores the original uploads as `source_main.png` and `source_ultrawide.png`, hashes them in the Stereo Processing Result manifest, and renders them under `Actual camera captures` before derived artifacts.
- Verified the real multipart upload path, source image delivery, health endpoint, persistent-service reinstall/restart, focused Python tests, and an unsigned generic-device iOS build. The current result remains diagnostic because `CALIBRATION_RATIONAL_MODEL_FAILS_HELDOUT_VALIDATION`; this limitation is displayed rather than hidden.
- Built, signed, and installed the updated `local.dev.DualCamStereoCapture2` app on the connected iPhone `Melt` after the single-port verification passed.
