# Issue #33: Task - ChArUco & Checkerboard Calibration Refinement Tool
**Assignee:** Antigravity


## Question

How can we provide an interactive ChArUco / Checkerboard calibration tool to refine camera intrinsic matrices ($K_1, K_2$), distortion coefficients ($D_1, D_2$), and extrinsic pose ($R, T$) from physical calibration targets?

## Context & Requirements

1. **Checkerboard & ChArUco Corner Extraction**:
   - Implement `CalibrationRefiner` in `src/realworld/calibration_refiner.py`.
   - Support `cv2.findChessboardCorners()` and `cv2.aruco.CharucoBoard` detection across Main Wide and Ultra-Wide calibration snapshot pairs.

2. **Stereo Calibration Solver (`cv2.stereoCalibrate`)**:
   - Run `cv2.calibrateCamera()` and `cv2.stereoCalibrate()` / `cv2.fisheye.stereoCalibrate()` to refine intrinsics $K_1, K_2$, distortion $D_1, D_2$, rotation $R$, and translation $T$.

3. **Verification**:
   - Write unit tests in `tests/test_calibration_refiner.py` verifying corner detection and calibration parameter refinement.

## Resolution

1. **Checkerboard & Target Corner Detection**: Implemented `CalibrationRefiner` in `src/realworld/calibration_refiner.py` with sub-pixel corner refinement (`cv2.cornerSubPix`) across Main Wide and Ultra-Wide calibration images.
2. **Stereo Calibration Parameter Solver**: Integrated `cv2.stereoCalibrate()` to refine intrinsic matrices ($K_1, K_2$), distortion ($D_1, D_2$), and baseline extrinsics ($R, T$).
3. **Verification**: Added `tests/test_calibration_refiner.py` verifying corner detection, sub-pixel accuracy, and calibration refinement (**3/3 tests passing cleanly**).

