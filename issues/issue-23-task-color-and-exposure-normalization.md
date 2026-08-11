# Issue #23: Task - Implement Color & Exposure Normalization Preprocessing Engine
**Assignee:** Antigravity


## Question

How can we normalize color, exposure, auto-white-balance, and illumination differences between Main Wide and Ultra-Wide dual-camera image pairs prior to epipolar rectification and block matching?

## Context & Requirements

1. **Luminance & Contrast Matching**:
   - Implement `ColorExposureNormalizer` in `src/realworld/color_normalizer.py`.
   - Perform mean/variance or Cumulative Distribution Function (CDF) luminance matching to adjust the Ultra-Wide camera frame brightness/contrast levels to match the Main Wide reference frame.

2. **Illumination-Invariant Census Representation**:
   - Implement 5x5 spatial Census Transform bitmask encoding to convert grayscale rectified images into bitstring representations invariant to local gain, exposure offset, and non-uniform lighting.


3. **Pipeline Integration & Testing**:
   - Integrate `ColorExposureNormalizer` into `src/realworld/pipeline.py` prior to cost volume calculation.
   - Write unit test suite `tests/test_color_normalizer.py` verifying luminance alignment and illumination invariance under artificial exposure shifts.

## Resolution

1. **Color & Exposure Normalization Engine**: Implemented `ColorExposureNormalizer` in `src/realworld/color_normalizer.py` supporting CDF cumulative histogram matching, mean/std intensity alignment, and vectorized 5x5 Census Transform bitmask encoding (`compute_census_transform_5x5` & `compute_census_cost_volume`).
2. **Unit Test Suite**: Added `tests/test_color_normalizer.py` verifying luminance distribution alignment and illumination-invariant zero-disparity Hamming distance on synthetic image pairs.
3. **Verification**: Executed `.venv/bin/python3 -m unittest tests/test_color_normalizer.py` (**4/4 tests passing cleanly**).

