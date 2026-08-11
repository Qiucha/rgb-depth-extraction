"""
Pipeline Quality Gate Exceptions and Validators.
Provides stage-specific error classes and validation functions for the stereo depth pipeline.
Each gate checks a measurable quality metric and raises a descriptive error when the threshold is violated.
"""

import numpy as np
from typing import Optional


class PipelineQualityError(RuntimeError):
    """Base class for pipeline quality gate failures."""
    pass


class CalibrationQualityError(PipelineQualityError):
    """Raised when calibration parameters fail quality checks."""
    pass


class RectificationQualityError(PipelineQualityError):
    """Raised when rectification output fails quality checks."""
    pass


class StereoMatchingQualityError(PipelineQualityError):
    """Raised when stereo matching produces insufficient valid disparities."""
    pass


class DepthQualityError(PipelineQualityError):
    """Raised when depth map contains insufficient valid depth values."""
    pass


def validate_calibration(R: np.ndarray, T: np.ndarray, reprojection_error: Optional[float] = None,
                          max_reprojection_error: float = 2.0):
    """
    Validates calibration parameters are physically plausible.

    :param R: 3x3 rotation matrix between cameras.
    :param T: Translation vector between cameras.
    :param reprojection_error: RMS reprojection error in pixels (if available).
    :param max_reprojection_error: Maximum acceptable reprojection error.
    :raises CalibrationQualityError: If calibration is invalid.
    """
    R_flat = np.array(R).reshape(3, 3)
    T_flat = np.array(T).flatten()

    # Check for identity rotation (placeholder calibration)
    if np.allclose(R_flat, np.eye(3), atol=1e-6):
        raise CalibrationQualityError(
            "Calibration R matrix is identity — this is a placeholder, not a real calibration. "
            "Run tools/calibrate_offline.py on real checkerboard images first."
        )

    # Check translation is non-zero
    baseline_norm = float(np.linalg.norm(T_flat))
    if baseline_norm < 1e-6:
        raise CalibrationQualityError(
            f"Calibration T vector is near-zero (norm={baseline_norm:.6f}). "
            "Stereo depth requires a non-zero baseline between cameras."
        )

    # Check reprojection error if available
    if reprojection_error is not None and reprojection_error > max_reprojection_error:
        raise CalibrationQualityError(
            f"Calibration reprojection error ({reprojection_error:.4f} px) exceeds threshold "
            f"({max_reprojection_error:.1f} px). Re-capture calibration images with better coverage."
        )


def validate_rectification(rect_left: np.ndarray, rect_right: np.ndarray,
                            min_valid_roi_fraction: float = 0.15):
    """
    Validates rectification output is not degenerate.

    :param rect_left: Rectified left image.
    :param rect_right: Rectified right image.
    :param min_valid_roi_fraction: Minimum fraction of non-black pixels required.
    :raises RectificationQualityError: If rectification output is degenerate.
    """
    if rect_left is None or rect_right is None:
        raise RectificationQualityError("Rectification produced None output.")

    if rect_left.size == 0 or rect_right.size == 0:
        raise RectificationQualityError("Rectification produced empty images.")

    # Check for mostly-black output (degenerate rectification)
    for label, img in [("left", rect_left), ("right", rect_right)]:
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img.astype(np.float64)
        nonblack_fraction = float(np.count_nonzero(gray > 5)) / max(gray.size, 1)
        if nonblack_fraction < min_valid_roi_fraction:
            raise RectificationQualityError(
                f"Rectified {label} image is {(1 - nonblack_fraction) * 100:.1f}% black "
                f"(valid ROI fraction {nonblack_fraction:.3f} < {min_valid_roi_fraction:.2f}). "
                "This indicates degenerate rectification — check calibration R and T."
            )


def validate_disparity(disparity: np.ndarray, min_fill_rate: float = 0.10):
    """
    Validates stereo matching produced sufficient valid disparities.

    :param disparity: 2D disparity map.
    :param min_fill_rate: Minimum fraction of pixels with valid disparity > 0.
    :raises StereoMatchingQualityError: If disparity fill rate is too low.
    """
    total_pixels = max(disparity.size, 1)
    valid_count = int(np.count_nonzero(disparity > 0))
    fill_rate = valid_count / total_pixels

    if fill_rate < min_fill_rate:
        raise StereoMatchingQualityError(
            f"Disparity fill rate ({fill_rate * 100:.1f}%, {valid_count}/{total_pixels} valid pixels) "
            f"is below threshold ({min_fill_rate * 100:.0f}%). "
            "This indicates stereo matching failed — likely due to poor rectification or "
            "featureless scene regions."
        )


def validate_depth(depth_map: np.ndarray, min_valid_rate: float = 0.05,
                    min_depth_m: float = 0.1, max_depth_m: float = 20.0):
    """
    Validates depth map contains physically plausible values.

    :param depth_map: 2D depth map in meters.
    :param min_valid_rate: Minimum fraction of pixels with plausible depth.
    :param min_depth_m: Minimum plausible depth in meters.
    :param max_depth_m: Maximum plausible depth in meters.
    :raises DepthQualityError: If depth map is mostly invalid.
    """
    total_pixels = max(depth_map.size, 1)
    valid_mask = (depth_map > min_depth_m) & (depth_map < max_depth_m) & np.isfinite(depth_map)
    valid_count = int(np.count_nonzero(valid_mask))
    valid_rate = valid_count / total_pixels

    if valid_rate < min_valid_rate:
        # Compute diagnostics
        finite_mask = np.isfinite(depth_map) & (depth_map > 0)
        if np.any(finite_mask):
            median_depth = float(np.median(depth_map[finite_mask]))
            min_actual = float(np.min(depth_map[finite_mask]))
            max_actual = float(np.max(depth_map[finite_mask]))
            diag = f" Actual range: [{min_actual:.3f}m, {max_actual:.3f}m], median: {median_depth:.3f}m."
        else:
            diag = " No finite positive depth values found at all."

        raise DepthQualityError(
            f"Valid depth rate ({valid_rate * 100:.1f}%, {valid_count}/{total_pixels} pixels in "
            f"[{min_depth_m}m, {max_depth_m}m]) is below threshold ({min_valid_rate * 100:.0f}%).{diag}"
        )
