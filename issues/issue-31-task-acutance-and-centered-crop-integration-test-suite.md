# Issue #31: Task - Acutance Enhancement & Centered Crop Master Pipeline Integration
**Assignee:** Antigravity


## Question

How can we integrate Lanczos remapping, acutance sharpening, and centered optical crop into the master pipeline and verify clean end-to-end depth extraction?

## Context & Requirements

1. **Master Pipeline Update**:
   - Update `run_realworld_pipeline()` in `src/realworld/pipeline.py` to invoke Lanczos remapping, `LanczosAcutanceEnhancer`, and principal-point centered Ultra-Wide crop.

2. **Visual Digest Studio Update**:
   - Regenerate Visual Digest telemetry and verify sharp image quality and clean disparity boundaries.

3. **Full Integration Test Suite**:
   - Execute full test suite (`python3 -m unittest discover -s tests`), verifying end-to-end pipeline execution and clean depth maps.

## Resolution

1. **Master Pipeline Integration**: Updated `run_realworld_pipeline()` in `src/realworld/pipeline.py` to apply Lanczos anti-aliased remapping, optical principal point centered crop, and `LanczosAcutanceEnhancer` sharpening.
2. **Telemetry & Visual Digest Studio**: Regenerated visual digest reports with high-acutance remapped frames and aligned FOV cropping.
3. **Verification**: Executed full unit and integration test suite (`.venv/bin/python3 -m unittest discover -s tests`), passing all **41/41 tests cleanly**.

