# Issue #30: Task - Centered Ultra-Wide Lens Crop & Physical Baseline Alignment Engine
**Assignee:** Antigravity


## Question

How can we center the Ultra-Wide camera FOV crop symmetrically on its optical principal point $(c_x, c_y)$ to prevent off-center shifting and maintain the true physical inter-camera baseline ($B \approx 19.5\text{mm}$)?

## Context & Requirements

1. **Optical Principal Point Centered Cropping**:
   - Update `HeterogeneousStereoRectifier` in `src/realworld/hetero_rectifier.py`.
   - Calculate exact optical center coordinates $(c_x^{UW}, c_y^{UW})$ from intrinsics $K2$ and compute FOV crop bounds centered symmetrically around $(c_x, c_y)$.
   - Eliminate off-center cropping errors that artificially enlarge the effective baseline distance and distort disparity search offsets.

2. **Projection Matrix $P2$ Baseline Locking**:
   - Verify rectified projection matrices $P1, P2$ match physical horizontal baseline $T_x = 19.5\text{mm}$.

3. **Testing & Verification**:
   - Write unit tests in `tests/test_centered_crop.py` verifying optical center alignment and baseline scaling.

## Resolution

1. **Optical Center Alignment Lock**: Updated `HeterogeneousStereoRectifier` in `src/realworld/hetero_rectifier.py` to enforce symmetric optical principal point alignment ($c_{x1} = c_{x2} = cx_{mean}$ and $c_{y1} = c_{y2} = cy_{mean}$), eliminating off-center FOV crop parallax.
2. **Physical Baseline Preservation**: Locked $P2[0, 3] = -P1[0, 0] \cdot B_{physical}$, strictly maintaining physical horizontal baseline $B = \|T_x\| \approx 19.5\text{mm}$ and zero disparity offset at infinity.
3. **Verification**: Added `tests/test_centered_crop.py` verifying principal point alignment and baseline scaling (**2/2 tests passing cleanly**).

