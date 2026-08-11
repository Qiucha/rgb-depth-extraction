Type: task
Status: open
Blocked by:

## Question

Fix the `float64` type crash and silent exception swallowing in `CalibrationRefiner` that prevents real-world calibration from succeeding.

### Root Cause

Three interacting bugs in [`calibration_refiner.py`](file:///Users/q/Documents/Projects/robotics_neo/src/realworld/calibration_refiner.py):

1. **float64 corner type crash** (line ~126, ~143): When `scale != 1.0` in `detect_corners`, dividing `float32` corners by a Python float implicitly upcasts to `float64`. Passing `float64` corners to `cv2.cornerSubPix`, `cv2.calibrateCamera`, or `cv2.stereoCalibrate` triggers a fatal C++ assertion: `cv2.error: (-215:Assertion failed) points.depth() == CV_32F`. Fix: cast back with `.astype(np.float32)` after rescaling.

2. **Silent exception swallowing** (lines 350-362): Stage 1 pre-calibration wraps `cv2.calibrateCamera` in a bare `except Exception: pass`. If it crashes (e.g., from the float64 bug), Stage 2's `cv2.stereoCalibrate` receives un-refined initial guesses with no warning, potentially producing wildly wrong results.

3. **Overflow in reprojection error** (line ~220-221): When `cv2.solvePnP` returns a degenerate pose, `cv2.projectPoints` projects corners to extreme coordinates. Squaring these overflows `float32`. The current guard checks `np.isfinite` after the fact, but the intermediate computation can still produce `RuntimeWarning: overflow encountered in square`.

### Acceptance Criteria

- [ ] `corners / scale` always returns `float32` — add `.astype(np.float32)` at both locations
- [ ] Replace bare `except Exception: pass` with specific error handling that logs a warning and falls through gracefully
- [ ] Reprojection error computation uses `float64` arithmetic throughout to prevent intermediate overflow
- [ ] Regression test: synthetic pair with `scale=1.5` path exercised, confirming no OpenCV assertion error
