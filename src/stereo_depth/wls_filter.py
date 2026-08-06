"""
Weighted Least Squares (WLS) and Edge-Preserving Filter module for disparity smoothing.
"""

import cv2
import numpy as np


class WLSDisparityFilter:
    def __init__(self, lambda_val=8000.0, sigma_val=1.5):
        """
        :param lambda_val: Regularization parameter for WLS filter (higher = smoother)
        :param sigma_val: Spatial color similarity sensitivity
        """
        self.lambda_val = lambda_val
        self.sigma_val = sigma_val

    def filter(self, raw_disparity, guide_image):
        """
        Applies edge-preserving smoothing to the disparity map guided by the RGB reference image.
        Uses OpenCV ximgproc WLS filter if available, or fast guided filter fallback.
        """
        if guide_image.ndim == 3:
            gray_guide = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_guide = guide_image.copy()

        # Check if cv2.ximgproc is available
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createDisparityWLSFilterGeneric'):
            wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
            wls_filter.setLambda(self.lambda_val)
            wls_filter.setSigmaColor(self.sigma_val)
            
            # WLS filter expects int16 formatted disparity (multiplied by 16)
            disp_int = (raw_disparity * 16.0).astype(np.int16)
            filtered_disp = wls_filter.filter(disp_int, gray_guide)
            return filtered_disp.astype(np.float32) / 16.0

        # High-performance Guided Filter fallback
        return self._guided_filter(raw_disparity, gray_guide, radius=8, eps=1e-2)

    def _guided_filter(self, p, I, radius=8, eps=1e-2):
        """
        Fast Guided Image Filter implementation in NumPy/OpenCV.
        I: Guidance image (grayscale float32)
        p: Input image to be filtered (disparity map float32)
        """
        I = I.astype(np.float32) / 255.0 if I.max() > 1.0 else I.astype(np.float32)
        p = p.astype(np.float32)
        ksize = (2 * radius + 1, 2 * radius + 1)

        N = cv2.boxFilter(np.ones_like(I), -1, ksize)
        mean_I = cv2.boxFilter(I, -1, ksize) / N
        mean_p = cv2.boxFilter(p, -1, ksize) / N
        mean_Ip = cv2.boxFilter(I * p, -1, ksize) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, -1, ksize) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, ksize) / N
        mean_b = cv2.boxFilter(b, -1, ksize) / N

        q = mean_a * I + mean_b
        return q
