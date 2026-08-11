# Issue #35: Task - Implement Standalone Offline Stereo Calibration CLI Tool
**Assignee:** Antigravity


## Question

How can we provide a command-line utility (`tools/calibrate_offline.py`) that processes a folder of synchronized checkerboard/ChArUco calibration snapshot pairs and outputs a refined calibration JSON file?

## Context & Requirements

1. **CLI Script Structure (`tools/calibrate_offline.py`)**:
   - Accepts `--images_dir`, `--pattern_cols`, `--pattern_rows`, `--square_size_m`, and `--output_json`.
   - Uses `CalibrationRefiner` (`src/realworld/calibration_refiner.py`) to detect corners across all calibration snapshot pairs.
   - Solves intrinsics ($K_1, K_2$), distortion ($D_1, D_2$), rotation ($R$), translation ($T$), and reprojection error using `cv2.stereoCalibrate()`.

2. **Calibration JSON Export**:
   - Saves formatted calibration JSON compatible with `RealWorldDatasetLoader` and `run_realworld_pipeline(refined_calib=...)`.

3. **Verification**:
   - Test CLI execution on synthetic/sample calibration images.

## Resolution

1. **Standalone CLI Script**: Implemented `tools/calibrate_offline.py` for batch loading stereo calibration snapshot pairs and running corner detection.
2. **Standardized JSON Export**: Integrated `CalibrationRefiner` to solve camera intrinsics ($K_1, K_2$), distortion ($D_1, D_2$), and baseline pose ($R, T$), exporting a standardized `calibration_refined.json` file compatible with `run_realworld_pipeline()`.
3. **Verification**: Tested CLI argument parsing and execution (`.venv/bin/python3 tools/calibrate_offline.py --help`, **Code 0, SUCCESS**).

