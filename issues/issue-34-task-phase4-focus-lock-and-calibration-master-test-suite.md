# Issue #34: Task - Phase 4 Focus Lock & Calibration Master Test Suite Verification
**Assignee:** Antigravity


## Question

How can we integrate focus-locked snapshot capture processing and refined intrinsic calibration into the master pipeline and verify end-to-end depth extraction?

## Context & Requirements

1. **Pipeline Integration**:
   - Update `run_realworld_pipeline()` in `src/realworld/pipeline.py` to support refined calibration parameters from `CalibrationRefiner`.

2. **Master Test Suite Verification**:
   - Execute full test suite (`python3 -m unittest discover -s tests`), verifying focus-locked capture decoding and accurate metric depth.

## Resolution

1. **Master Pipeline Calibration Override**: Updated `run_realworld_pipeline()` in `src/realworld/pipeline.py` to support optional `refined_calib` override parameters from `CalibrationRefiner`.
2. **Focus-Locked Capture Verification**: Verified focus-locked snapshot decoding and unblurred frame alignment across Main Wide and Ultra-Wide sensors.
3. **Verification**: Executed full unit and integration test suite (`.venv/bin/python3 -m unittest discover -s tests`), passing all **44/44 tests cleanly**.

