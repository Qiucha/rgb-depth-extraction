"""
Tests for pipeline quality gates — calibration, rectification, disparity, and depth validators.
"""

import numpy as np
import pytest

from src.realworld.quality_gates import (
    CalibrationQualityError,
    RectificationQualityError,
    StereoMatchingQualityError,
    DepthQualityError,
    validate_calibration,
    validate_rectification,
    validate_disparity,
    validate_depth,
)


class TestCalibrationGate:
    def test_identity_R_raises(self):
        """Identity rotation should be rejected as a placeholder."""
        with pytest.raises(CalibrationQualityError, match="identity"):
            validate_calibration(np.eye(3), np.array([19.5, 0, 0]))

    def test_zero_T_raises(self):
        """Zero translation should be rejected."""
        R = np.array([[0.9998, -0.01, 0.005], [0.01, 0.9999, -0.003], [-0.005, 0.003, 0.9998]])
        with pytest.raises(CalibrationQualityError, match="near-zero"):
            validate_calibration(R, np.array([0, 0, 0]))

    def test_high_reprojection_error_raises(self):
        """Reprojection error above threshold should be rejected."""
        R = np.array([[0.9998, -0.01, 0.005], [0.01, 0.9999, -0.003], [-0.005, 0.003, 0.9998]])
        T = np.array([19.5, 0, 0])
        with pytest.raises(CalibrationQualityError, match="reprojection error"):
            validate_calibration(R, T, reprojection_error=5.0, max_reprojection_error=2.0)

    def test_valid_calibration_passes(self):
        """Good calibration should pass without error."""
        R = np.array([[0.9998, -0.01, 0.005], [0.01, 0.9999, -0.003], [-0.005, 0.003, 0.9998]])
        T = np.array([19.5, 0.5, -0.2])
        validate_calibration(R, T, reprojection_error=0.5)  # Should not raise


class TestRectificationGate:
    def test_mostly_black_raises(self):
        """Mostly-black rectified images should be rejected."""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        normal = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        with pytest.raises(RectificationQualityError, match="black"):
            validate_rectification(black, normal)

    def test_valid_rectification_passes(self):
        """Normal images should pass."""
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        validate_rectification(img, img)  # Should not raise


class TestDisparityGate:
    def test_mostly_zero_disparity_raises(self):
        """Disparity map with < 10% valid pixels should be rejected."""
        disp = np.zeros((480, 640), dtype=np.float32)
        disp[0, 0] = 5.0  # One pixel valid
        with pytest.raises(StereoMatchingQualityError, match="fill rate"):
            validate_disparity(disp)

    def test_valid_disparity_passes(self):
        """Disparity with good fill rate should pass."""
        disp = np.random.uniform(1, 50, (480, 640)).astype(np.float32)
        validate_disparity(disp)  # Should not raise


class TestDepthGate:
    def test_mostly_invalid_depth_raises(self):
        """Depth map with mostly infinity/zero should be rejected."""
        depth = np.zeros((480, 640), dtype=np.float32)
        with pytest.raises(DepthQualityError, match="Valid depth rate"):
            validate_depth(depth)

    def test_valid_depth_passes(self):
        """Depth map with plausible values should pass."""
        depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
        validate_depth(depth)  # Should not raise
