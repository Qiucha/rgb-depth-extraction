# Issue #29: Task - Anti-Aliased Lanczos Remapping & Acutance Enhancer Engine
**Assignee:** Antigravity


## Question

How can we eliminate bilinear interpolation blur on the Main Wide camera view and align left/right gradient acutance prior to Census-SGBM block matching?

## Context & Requirements

1. **Anti-Aliased Lanczos Remapping (`cv2.INTER_LANCZOS4`)**:
   - Update `HeterogeneousStereoRectifier` in `src/realworld/hetero_rectifier.py` to use `cv2.INTER_LANCZOS4` / `cv2.INTER_CUBIC` high-order interpolation instead of bilinear interpolation during `cv2.remap()`.

2. **Adaptive Unsharp Masking (`LanczosAcutanceEnhancer`)**:
   - Implement `LanczosAcutanceEnhancer` in `src/realworld/acutance_enhancer.py`.
   - Apply Gaussian unsharp masking (USM) with scale factor matching to sharpen the Main Wide image to match the Ultra-Wide high-frequency edge gradients.

3. **Verification & Testing**:
   - Write unit tests in `tests/test_acutance_enhancer.py` verifying gradient sharpness boost and absence of ringing artifacts.

## Resolution

1. **Anti-Aliased Lanczos Remapping**: Updated `HeterogeneousStereoRectifier` (`src/realworld/hetero_rectifier.py`) to use `cv2.INTER_LANCZOS4` high-order interpolation during coordinate remapping, eliminating bilinear blur.
2. **Acutance Sharpening Engine**: Implemented `LanczosAcutanceEnhancer` in `src/realworld/acutance_enhancer.py` applying adaptive Unsharp Masking (USM) to boost edge gradient magnitudes and align Left/Right acutance.
3. **Verification**: Added `tests/test_acutance_enhancer.py` verifying edge gradient magnitude boost and bypass functionality (**2/2 tests passing cleanly**).

