Type: task
Status: open
Blocked by:

## Question

Fix the resolution mismatch in `calibrate_offline.py` that corrupts K2 intrinsics when Main and Ultra-Wide images have different native resolutions.

### Root Cause

In [`calibrate_offline.py`](file:///Users/q/Documents/Projects/robotics_neo/tools/calibrate_offline.py) (line 41-42), when `img_m` and `img_u` have different resolutions (which they always do — Main is higher-res than Ultra-Wide on iPhone), the ultra-wide image is resized to match Main's resolution *before* calibration:

```python
if img_m.shape[:2] != img_u.shape[:2]:
    img_u = cv2.resize(img_u, (img_m.shape[1], img_m.shape[0]), ...)
```

As a result, the calibrated `K2` and `D2` correspond to the *Main camera's* resolution, not the Ultra-Wide's native resolution. When `pipeline.py` later loads the Ultra-Wide image at its *original* resolution and passes it to `hetero_rectifier.py`, the rectifier scales `K2` again assuming it was calibrated at the original UW resolution — producing a wildly incorrect focal length and principal point.

### Acceptance Criteria

- [ ] `calibrate_offline.py` records the original Ultra-Wide resolution before resizing
- [ ] After calibration, `K2` is scaled back from Main resolution to the original Ultra-Wide resolution before export to JSON
- [ ] `calibration_refined.json` includes `image_size_main` and `image_size_ultrawide` fields documenting which resolution each K was calibrated at
- [ ] Regression test: calibrate with different-resolution synthetic pairs, verify `K2` focal length matches Ultra-Wide native resolution
