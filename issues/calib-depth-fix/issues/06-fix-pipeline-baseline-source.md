Type: task
Status: open
Blocked by: 05

## Question

Fix the pipeline's depth calculation using stale `frame_data["T"]` (from metadata.json placeholder) instead of the refined calibration T, producing incorrect metric depth scale.

### Root Cause

In [`pipeline.py`](file:///Users/q/Documents/Projects/robotics_neo/src/realworld/pipeline.py) (line 131):

```python
baseline_m = float(np.linalg.norm(frame_data["T"])) / 1000.0
```

This reads the translation from the *original* `frame_data` (loaded from `metadata.json`), not from the refined calibration `T` variable that was actually used for rectification (lines 84-89). The `metadata.json` contains placeholder values (`T = [19.5, 0, 0]` mm with identity rotation), while the refined calibration JSON may contain the actual solved translation.

Additionally, the unit conversion `/ 1000.0` assumes T is always in millimeters, but there's no explicit unit documentation — if a calibration produces T in meters, this will divide by 1000 again.

### Acceptance Criteria

- [ ] Line 131 reads from the `T` variable (the refined/active T), not `frame_data["T"]`
- [ ] `calibration_refined.json` documents T units explicitly (a `"T_units": "mm"` or `"T_units": "m"` field)
- [ ] Pipeline logs the baseline value it's using: `[Pipeline] Using baseline: XX.XX mm (YY.YYYY m)`
- [ ] Regression test: verify `DepthCalculator` receives the correct baseline from refined calibration, not the placeholder
