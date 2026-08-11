# Issue #36: Task - Offline Calibration CLI Tool Unit Test Suite & Pipeline Verification
**Assignee:** Antigravity


## Question

How can we verify the offline stereo calibration CLI tool and ensure exported JSON calibration matrices load seamlessly into the master pipeline?

## Context & Requirements

1. **Unit Test Suite (`tests/test_calibrate_offline_cli.py`)**:
   - Verify CLI parsing, checkerboard corner extraction, calibration solving, and JSON export.

2. **Master Test Suite Verification**:
   - Run full test suite (`python3 -m unittest discover -s tests`), verifying pipeline integration with refined offline calibration files.

## Resolution

1. **CLI & Calibration Integration Tests**: Added `tests/test_calibrate_offline_cli.py` verifying CLI batch snapshot processing, sub-pixel corner detection, `cv2.stereoCalibrate()` solving, and JSON matrix export.
2. **Master Pipeline Override Verification**: Tested passing exported JSON parameters directly into `run_realworld_pipeline(refined_calib=...)`.
3. **Verification**: Executed full unit and integration test suite (`.venv/bin/python3 -m unittest discover -s tests`), passing all **47/47 tests cleanly**.

