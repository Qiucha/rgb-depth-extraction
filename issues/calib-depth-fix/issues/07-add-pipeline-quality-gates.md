Type: task
Status: open
Blocked by: 06

## Question

Add quality gates at each pipeline stage that fail loudly (raise or log critical warnings) instead of silently passing garbage downstream.

### Root Cause

The pipeline runs as a straight line with no intermediate validation:
1. **Calibration** → no check that reprojection error is below threshold, no check that R ≠ Identity
2. **Rectification** → no check that epipolar lines are horizontal, no check for degenerate rectification (all-black output, valid ROI too small)
3. **Stereo Matching** → no check for disparity fill rate (% of pixels with valid disparity > 0)
4. **Depth** → no check for depth plausibility (values in reasonable range, not all infinity/NaN)

### Acceptance Criteria

- [ ] **Calibration gate**: If `calibration_refined.json` has `R = Identity` or reprojection error > 2.0px, raise `CalibrationQualityError` with actionable message
- [ ] **Rectification gate**: After remapping, sample epipolar alignment by checking that corresponding feature points in left/right images have |Δy| < 2px. Log warning if valid ROI covers < 30% of image area
- [ ] **Disparity gate**: After stereo matching, check `disparity_fill_rate = count(disparity > 0) / total_pixels`. If < 10%, raise `StereoMatchingQualityError`
- [ ] **Depth gate**: After depth conversion, check `valid_depth_rate = count(0.1m < depth < 20m) / total_pixels`. If < 5%, raise `DepthQualityError`
- [ ] All gates are toggleable via a `strict_quality_gates: bool = True` pipeline parameter (so existing tests don't break)
- [ ] Gate failures include diagnostic info: the actual metric value and the threshold
