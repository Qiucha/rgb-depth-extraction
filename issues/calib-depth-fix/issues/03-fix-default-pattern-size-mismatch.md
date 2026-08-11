Type: task
Status: open
Blocked by:

## Question

Fix the default pattern size and square size in `calibrate_offline.py` to match the real calibration target (5×5 inner corners, 0.26m squares on iPad).

### Root Cause

In [`calibrate_offline.py`](file:///Users/q/Documents/Projects/robotics_neo/tools/calibrate_offline.py) (lines 90-92), the defaults are:
- `--pattern_cols 9` and `--pattern_rows 6` (expecting a 10×7 grid with 9×6 inner corners)
- `--square_size_m 0.025` (25mm squares)

The actual calibration target is a **6×6 grid displayed on iPad** with **5×5 inner corners** and **0.26m square size**. The auto-discovery mechanism searches a fixed list of candidate patterns (line 43-47 in `calibration_refiner.py`) that does not include `(5, 5)`.

This means:
1. If the user runs `calibrate_offline.py` without `--pattern_cols 5 --pattern_rows 5 --square_size_m 0.26`, corner detection fails on every image.
2. Even with auto-discovery enabled, `(5, 5)` is never tried because it's not in the candidate list.
3. The fallback returns `R = Identity` and `T = [-0.0195, 0, 0]` — the placeholder values.

### Acceptance Criteria

- [ ] `(5, 5)` added to the auto-discovery candidate pattern list in `CalibrationRefiner`
- [ ] Default `--square_size_m` changed to `0.26` (or at minimum, the CLI prints a warning if using the old default with the real calibration images)
- [ ] Default `--pattern_cols` and `--pattern_rows` changed to `5` and `5` to match the actual target
- [ ] A `--pattern_type` flag or config file records the target spec so it doesn't need to be re-entered
