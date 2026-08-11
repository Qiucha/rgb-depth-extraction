# Issue #40: Task - Census & Normalized Gradient Pre-Filtering for Photometric Resilience

## Question
How do we enhance pre-filtering in `CensusSGBMMatcher` and `ColorExposureNormalizer` using normalized gradient / Census bitmasks to eliminate independent ISP local tone mapping and photometric intensity discrepancies?

## Resolution
Added `ColorExposureNormalizer.compute_normalized_gradient_image()` in `src/realworld/color_normalizer.py` computing variance-normalized Sobel magnitude images. Updated `CensusSGBMMatcher.compute_disparity()` in `src/realworld/census_sgbm_matcher.py` to feed normalized gradient pre-filtered images to `cv2.StereoSGBM` dynamic programming. Verified with `tests/test_census_gradient_prefilter.py` demonstrating robust matching under severe (2x) photometric tone curve mismatch (**50/50 tests passing**).
