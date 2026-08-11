Type: task
Status: open
Blocked by: 07

## Question

Validate the complete fixed pipeline end-to-end using the real iPhone 17 calibration images in `data/calib_snapshots/` and real scene stereo pairs, confirming it produces a plausible depth map.

### What This Ticket Does

This is the final validation — not a code change, but a verification that all upstream fixes work together:

1. Run `tools/calibrate_offline.py --images_dir data/calib_snapshots --pattern_cols 5 --pattern_rows 5 --square_size_m 0.26`
2. Verify the output `calibration_refined.json` has:
   - R ≠ Identity (actual rotation between cameras)
   - Reprojection error < 1.0 px
   - Physically reasonable baseline (~10-25mm for iPhone dual cameras)
3. Run the pipeline on a real scene stereo pair with the refined calibration
4. Verify the visual digest shows:
   - Rectified images with horizontal epipolar grid alignment
   - Non-trivial disparity map (> 20% fill rate)
   - Depth map with physically plausible values (0.3m - 10m for indoor scenes)
5. Visual inspection of `calib_rectified_verification.jpg` — epipolar lines should align across left/right

### Acceptance Criteria

- [ ] `calibrate_offline.py` runs without errors on real `data/calib_snapshots/`
- [ ] Reprojection error < 1.0 px
- [ ] `calib_rectified_verification.jpg` shows aligned epipolar lines
- [ ] Pipeline produces a depth map with > 20% valid pixel fill rate
- [ ] Depth values are in the physically plausible range for the scene
- [ ] All quality gates pass (no `CalibrationQualityError`, `StereoMatchingQualityError`, or `DepthQualityError`)
