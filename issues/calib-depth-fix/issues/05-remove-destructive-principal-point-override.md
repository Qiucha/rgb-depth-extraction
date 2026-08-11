Type: task
Status: open
Blocked by: 04

## Question

Remove or fix the destructive manual principal point override that happens *after* `cv2.stereoRectify` in `hetero_rectifier.py`, which breaks the carefully computed rectification geometry.

### Root Cause

In [`hetero_rectifier.py`](file:///Users/q/Documents/Projects/robotics_neo/src/realworld/hetero_rectifier.py) (lines 124-134 for fisheye, lines 153-163 for pinhole), after `cv2.stereoRectify` computes P1 and P2 (which already align cy when `CALIB_ZERO_DISPARITY` is used), the code manually overwrites them:

```python
cy_mean = (P1[1, 2] + P2[1, 2]) / 2.0
cx_mean = (P1[0, 2] + P2[0, 2]) / 2.0
P1[1, 2] = cy_mean
P2[1, 2] = cy_mean
P1[0, 2] = cx_mean
P2[0, 2] = cx_mean
P2[0, 3] = -P1[0, 0] * baseline_val
```

**Problems:**
1. `CALIB_ZERO_DISPARITY` already enforces `P1[1,2] == P2[1,2]` — the cy override is redundant at best.
2. Overwriting `cx` in both P1 and P2 to the same value destroys the disparity offset (`doffs = cx_right - cx_left`) that OpenCV computed. With `P1[0,2] == P2[0,2]`, the images are shifted to the same horizontal center, which can crop valid image regions and create artificial zero-disparity.
3. Overwriting `P2[0,3]` with `-f * baseline_val` ignores the projection that `stereoRectify` computed, which accounts for the actual geometry including any non-horizontal translation component.

### Acceptance Criteria

- [ ] Remove manual cx/cy override — trust `stereoRectify` output with `CALIB_ZERO_DISPARITY`
- [ ] Remove manual `P2[0,3]` override — let OpenCV's computed projection matrix stand
- [ ] If the Q matrix needs to be rebuilt, derive it from the actual P1 and P2 values, not from overridden values
- [ ] Regression test: verify `doffs = P2[0,2] - P1[0,2]` is correctly computed and non-zero for heterogeneous cameras
