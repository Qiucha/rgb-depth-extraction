# Issue #43: Task - PSF Gradient Mismatch & Lanczos Scale Factor Optimization

## Question
How do we optimize high-frequency gradient matching across heterogeneous sensors (Main vs Ultra-Wide) by tuning Lanczos anti-aliasing interpolation scale factors and smoothing pre-filters?

## Resolution
Implemented `PSFGradientOptimizer` in `src/realworld/psf_gradient_optimizer.py`. Measures 2D Sobel gradient magnitude distributions between Main and Ultra-Wide views, estimates MTF/PSF gradient mismatch ratio $\gamma = \frac{\bar{G}_{\text{main}}}{\bar{G}_{\text{uw}}}$, and adaptively applies proportional Gaussian pre-filtering and `LanczosAcutanceEnhancer` high-frequency boosting to equalize sensor PSFs prior to Census-SGBM matching. Verified with `tests/test_psf_gradient_optimizer.py` (**59/59 master tests passing**).
