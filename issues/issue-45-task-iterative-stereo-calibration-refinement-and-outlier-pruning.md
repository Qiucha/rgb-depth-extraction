# Issue #45: Task - Iterative Stereo Calibration Refinement, Pattern Auto-Discovery & Outlier Pruning

## Question
How do we enhance `CalibrationRefiner` and `tools/calibrate_offline.py` to fix corner detection unpacking errors, implement automatic pattern dimension discovery, perform multi-stage corner extraction, and execute iterative outlier pruning to ensure robust sub-pixel stereo camera calibration on captured image pairs?

## Resolution
Refactored `CalibrationRefiner` and `tools/calibrate_offline.py` with automatic checkerboard grid pattern discovery, fixed `cv2.findChessboardCornersSB` return tuple unpacking, added multi-stage image pyramid preprocessing, and implemented an iterative 2-pass stereo calibration outlier pruning loop. Discovered (5, 4) target in `data/calib_snapshots`, pruned high reprojection error pairs, and achieved refined sub-pixel stereo calibration.
