# Issue #41: Task - Expanded Disparity Range (numDisparities up to 160) & Cost Volume Tuning

## Question
How do we configure `CensusSGBMMatcher` and the depth extraction pipeline to support configurable expanded search ranges (`numDisparities` = 128 / 160) and inspect raw cost volume bounds to eliminate false min-cost matches on near objects?

## Resolution
Configured `CensusSGBMMatcher` in `src/realworld/census_sgbm_matcher.py` to support expanded disparity search ranges up to 160px (`max_disparity=160`, aligned to multiples of 16 for OpenCV SGBM). Added diagnostic helper `compute_cost_volume_bounds()` calculating Hamming cost volume statistics and boundary saturation ratio to detect out-of-bounds disparity candidates. Verified with `tests/test_expanded_disparity_range.py` (**51/51 tests passing**).
