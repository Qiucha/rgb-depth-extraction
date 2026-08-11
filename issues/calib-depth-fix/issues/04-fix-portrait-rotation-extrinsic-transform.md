Type: task
Status: open
Blocked by: 01, 02, 03

## Question

Fix the portrait-to-landscape auto-rotation in `hetero_rectifier.py` that transforms image data and intrinsics but leaves the extrinsic matrices R and T unchanged — causing "double rotation" that destroys rectification.

### Root Cause

In [`hetero_rectifier.py`](file:///Users/q/Documents/Projects/robotics_neo/src/realworld/hetero_rectifier.py) (lines 75-81), when portrait images (H > W) are detected, the code:
1. ✅ Rotates both images 90° CCW
2. ✅ Transforms K1 and K2 intrinsics via `_rotate_intrinsics_ccw90`
3. ❌ Leaves R and T **completely unchanged**

With the original T describing a vertical baseline (because the phone is portrait), passing unrotated T to `cv2.stereoRectify` causes OpenCV to attempt a *second* 90° rotation to force the baseline onto the X-axis. This "double rotation" produces heavily sheared or blank rectified images.

### Fix

After rotating images and intrinsics, apply the same 90° CCW coordinate transform to R and T:

```python
R_z90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
R = R_z90 @ R @ R_z90.T
T = R_z90 @ T.reshape(3, 1)
```

### Acceptance Criteria

- [ ] R and T are transformed consistently with the image rotation in `rectify_pair`
- [ ] Regression test: portrait synthetic stereo pair → rectify → verify epipolar lines are horizontal
- [ ] No behavior change for landscape inputs (rotation path not entered)
