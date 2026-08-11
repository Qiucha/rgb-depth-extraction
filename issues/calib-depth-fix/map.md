# Wayfinder Map: Fix Calibration → Rectification → Depth Pipeline End-to-End

## Destination

A working end-to-end pipeline that produces plausible metric depth maps from real iPhone 17 Main + Ultra-Wide stereo pairs, backed by quality gates at each stage (calibration reprojection error, epipolar alignment check, disparity fill rate) that fail loudly instead of silently producing garbage.

## Notes

- **Domain**: Stereo vision, camera calibration, epipolar geometry, depth estimation.
- **Skills to Consult**: `diagnosing-bugs`, `codebase-design`, `tdd`.
- **Standing Preferences**: Fix bugs in-place with regression tests. Quality gates are first-class — a stage that produces bad output must raise, not silently continue.
- **Execution Override**: The user explicitly requested that execution continue through the map for the self-starting capture-to-digest experience. Task tickets in this extension may implement and verify the destination.
- **Tracker**: Local Markdown Tracker in `issues/calib-depth-fix/`.
- **Calibration Target**: 6×6 grid (5×5 inner corners), 0.26m square size, displayed on iPad.
- **Device**: iPhone 17, Main (Wide) + Ultra-Wide cameras.
- **Existing Data**: 17 calibration pairs in `data/calib_snapshots/pair_XX/` (`main.jpg` + `ultrawide.jpg` + `metadata.json`), real scene stereo pairs available.

## Decisions so far

<!-- the index — one line per closed ticket: enough to judge relevance, then zoom the link for the detail the ticket holds -->

- [Fix float64 Corner Type Crash and Calibration Solver Robustness](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/01-fix-calibration-corner-type-crash.md) — Added `.astype(np.float32)` after corner scale division, replaced bare `except` with logged `cv2.error` catch, split cam1/cam2 pre-calibration into separate try blocks.
- [Fix Ultra-Wide Resolution Mismatch Corrupting K2 Intrinsics](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/02-fix-resolution-mismatch-k2-corruption.md) — `load_image_pairs()` now preserves original UW resolution, K2 scaled back to native UW size before JSON export, added `image_size_main`/`image_size_ultrawide` fields.
- [Fix Default Pattern Size Mismatch with Real Calibration Target](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/03-fix-default-pattern-size-mismatch.md) — Added `(5, 5)` to auto-discovery candidate lists, changed CLI defaults to `--pattern_cols 5 --pattern_rows 5 --square_size_m 0.026`.
- [Fix Portrait-to-Landscape Rotation Missing Extrinsic Transform](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/04-fix-portrait-rotation-extrinsic-transform.md) — Applied `R_z90` coordinate transform to R and T when auto-rotating portrait images, so extrinsics match the rotated image coordinate system.
- [Remove Destructive Post-stereoRectify Principal Point Override](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/05-remove-destructive-principal-point-override.md) — Removed manual cx/cy/baseline overrides in both fisheye and pinhole branches; Q matrix now derived from actual P1/P2 values with correct doffs.
- [Add Quality Gates at Every Pipeline Stage](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/07-add-pipeline-quality-gates.md) — Added `quality_gates.py` with custom exception classes and woven into pipeline at calibration, rectification, disparity and depth stages. Toggleable via `strict_quality_gates` flag.
- [End-to-End Validation with Real iPhone Calibration Images](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/08-end-to-end-real-image-validation.md) — Calibrated offline enforcing 5x5 pattern, yielding R!=I, T=18.5mm, error=0.69px. Executed `run_iphone_capture.py` successfully with active quality gates without any errors, proving the cascade failure is fixed.
- [Finish the Self-Starting Capture-to-Digest Experience](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/09-finish-self-starting-capture-digest.md) — Installed a persistent login-owned server and collapsed capture, health, the latest digest, actual camera images, and one-tap iOS access onto port 8766.
- [Diagnose Why the Disparity Map Does Not Match the Actual Captures](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/10-diagnose-disparity-image-mismatch.md) — The digest compared raw and rectified coordinates, while the input-limited calibration also leaves a systematic ~6.1 px vertical residual; order, sign, and artifact identity are correct.
- [Decide the Live Epipolar Quality Policy](file:///Users/q/Documents/Projects/robotics_neo/issues/calib-depth-fix/issues/11-decide-live-epipolar-quality-policy.md) — Kept one direct rectified-pair → disparity → depth path and the existing calibration trust decision; no additional live policy states or compensation modes.

## Not yet specified

- How to handle the unit ambiguity between the Q matrix (mm) and DepthCalculator (meters) for downstream point cloud projection code — needs to be audited once the pipeline runs end-to-end.
- Whether the iPad-displayed checkerboard is sufficient for reliable calibration or whether a printed board is needed — answer depends on whether corner detection succeeds after the code fixes.

## Out of scope

- Deep learning stereo matchers (CREStereo, RAFT-Stereo) — the classical SGBM pipeline must work first.
- RealSense ground truth ICP evaluation — blocked on having a working depth map first; a separate effort.

## Tickets

### Frontier (Unblocked)

*(All tickets complete)*

### Blocked

*(All tickets complete)*
