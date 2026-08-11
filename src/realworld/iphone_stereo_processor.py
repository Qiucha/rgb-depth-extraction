"""One limitation-aware numerical core for iPhone Main + Ultra-Wide pairs."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from .stereo_contracts import (
    CalibrationResult,
    CalibrationStatus,
    ContractError,
    DisparityResult,
    MatcherMethod,
    MetricDepthResult,
    PixelSize,
    ProcessedStereoResult,
    ProcessingDisposition,
    QuarterTurn,
    RectificationGeometry,
    RectifiedStereoPair,
    SourceOrientations,
)


def _orient(image: np.ndarray, turn: QuarterTurn) -> np.ndarray:
    if turn is QuarterTurn.IDENTITY:
        return image
    if turn is QuarterTurn.CLOCKWISE_90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turn is QuarterTurn.ROTATE_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _orientation_to_native(
    image: np.ndarray,
    native_size: PixelSize,
    camera_name: str,
) -> QuarterTurn:
    source_size = image.shape[1], image.shape[0]
    if source_size == native_size.cv:
        return QuarterTurn.IDENTITY
    if source_size == (native_size.height, native_size.width):
        return QuarterTurn.COUNTERCLOCKWISE_90
    raise ContractError(
        f"{camera_name} source size {source_size} does not match calibration-native "
        f"{native_size.cv} in landscape or portrait orientation"
    )


class IPhoneStereoRectifier:
    """Physical-left-first rectifier with maps compiled in native source coordinates."""

    def __init__(
        self,
        calibration_result: CalibrationResult,
        output_size: PixelSize,
    ) -> None:
        self.calibration_result = calibration_result
        self.calibration = calibration_result.processing_geometry
        self.output_size = output_size
        self._plans: dict[SourceOrientations, tuple[Any, ...]] = {}

    def _compile(self, orientations: SourceOrientations) -> tuple[Any, ...]:
        if orientations in self._plans:
            return self._plans[orientations]
        calibration = self.calibration
        if calibration.main_native_size != calibration.ultrawide_native_size:
            raise ContractError("Current OpenCV rectification requires equal native canvas sizes")

        # Stored transform maps Main coordinates into Ultra-Wide coordinates. OpenCV
        # receives physical left Ultra-Wide first, so invert it to Ultra-Wide -> Main.
        rotation_left_to_right = calibration.main_to_ultrawide_rotation.T
        translation_left_to_right = (
            -rotation_left_to_right
            @ calibration.main_to_ultrawide_translation_m.reshape(3, 1)
        )
        native_size = calibration.main_native_size.cv
        distortion_left = calibration.ultrawide_distortion
        distortion_right = calibration.main_distortion
        flags = cv2.CALIB_ZERO_DISPARITY
        R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
            calibration.ultrawide_matrix,
            distortion_left,
            calibration.main_matrix,
            distortion_right,
            native_size,
            rotation_left_to_right,
            translation_left_to_right,
            flags=flags,
            alpha=0,
            newImageSize=self.output_size.cv,
        )
        left_x, left_y = cv2.initUndistortRectifyMap(
            calibration.ultrawide_matrix,
            distortion_left,
            R_left,
            P_left,
            self.output_size.cv,
            cv2.CV_32FC1,
        )
        right_x, right_y = cv2.initUndistortRectifyMap(
            calibration.main_matrix,
            distortion_right,
            R_right,
            P_right,
            self.output_size.cv,
            cv2.CV_32FC1,
        )
        margin = 3.0
        left_valid = (
            (left_x >= margin)
            & (left_y >= margin)
            & (left_x < calibration.ultrawide_native_size.width - margin)
            & (left_y < calibration.ultrawide_native_size.height - margin)
        )
        right_valid = (
            (right_x >= margin)
            & (right_y >= margin)
            & (right_x < calibration.main_native_size.width - margin)
            & (right_y < calibration.main_native_size.height - margin)
        )
        focal = float(P_left[0, 0])
        signed_baseline = -float(P_right[0, 3]) / focal
        if signed_baseline <= 0:
            raise ContractError(
                "Physical-left-first rectification did not produce positive disparity geometry"
            )
        geometry = RectificationGeometry(
            left_projection=P_left,
            right_projection=P_right,
            q_matrix=Q,
            focal_length_px=focal,
            baseline_m=signed_baseline,
            disparity_offset_px=float(P_right[0, 2] - P_left[0, 2]),
        )
        plan = left_x, left_y, right_x, right_y, left_valid, right_valid, geometry
        self._plans[orientations] = plan
        return plan

    def rectify(
        self,
        main: np.ndarray,
        ultrawide: np.ndarray,
        orientations: SourceOrientations,
    ) -> RectifiedStereoPair:
        main_native = _orient(main, orientations.main_to_calibration_native)
        ultrawide_native = _orient(
            ultrawide, orientations.ultrawide_to_calibration_native
        )
        expected_main = self.calibration.main_native_size.cv
        expected_ultrawide = self.calibration.ultrawide_native_size.cv
        if (main_native.shape[1], main_native.shape[0]) != expected_main:
            raise ContractError(
                f"Main source size {(main_native.shape[1], main_native.shape[0])} "
                f"does not match calibration-native {expected_main}"
            )
        if (ultrawide_native.shape[1], ultrawide_native.shape[0]) != expected_ultrawide:
            raise ContractError(
                f"Ultra-Wide source size {(ultrawide_native.shape[1], ultrawide_native.shape[0])} "
                f"does not match calibration-native {expected_ultrawide}"
            )
        left_x, left_y, right_x, right_y, left_valid, right_valid, geometry = (
            self._compile(orientations)
        )
        left = cv2.remap(
            ultrawide_native,
            left_x,
            left_y,
            cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
        )
        right = cv2.remap(
            main_native,
            right_x,
            right_y,
            cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
        )
        joint = left_valid & right_valid
        diagnostics = {
            "joint_valid_fraction": float(np.mean(joint)),
            "left_grayscale_std": float(np.std(_gray(left)[left_valid])),
            "right_grayscale_std": float(np.std(_gray(right)[right_valid])),
            "physical_left": "ultrawide",
            "physical_right": "main",
            "disparity_convention": "x_left_minus_x_right",
        }
        return RectifiedStereoPair(
            left_ultrawide=left,
            right_main=right,
            left_validity=left_valid,
            right_validity=right_valid,
            joint_validity=joint,
            geometry=geometry,
            diagnostics=diagnostics,
        )


class EvidenceSlidingWindowMatcher:
    PROFILE_VERSION = "sliding-window-31-v1"

    def __init__(self, min_disparity: int = 0, max_disparity: int = 64) -> None:
        self.window_size = 31
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity

    def _cost_volume(
        self, reference: np.ndarray, source: np.ndarray, reverse: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        left = _gray(reference)
        right = _gray(source)
        mean_left = cv2.boxFilter(left, -1, (self.window_size, self.window_size))
        left_zero = left - mean_left
        left_var = cv2.boxFilter(left_zero * left_zero, -1, (self.window_size, self.window_size))
        costs = []
        for disparity in range(self.min_disparity, self.max_disparity):
            shifted = np.zeros_like(right)
            if disparity == 0:
                shifted[:] = right
            elif reverse:
                shifted[:, :-disparity] = right[:, disparity:]
            else:
                shifted[:, disparity:] = right[:, :-disparity]
            mean_right = cv2.boxFilter(
                shifted, -1, (self.window_size, self.window_size)
            )
            right_zero = shifted - mean_right
            numerator = cv2.boxFilter(
                left_zero * right_zero, -1, (self.window_size, self.window_size)
            )
            right_var = cv2.boxFilter(
                right_zero * right_zero, -1, (self.window_size, self.window_size)
            )
            costs.append(
                numerator / np.sqrt(np.maximum(left_var * right_var, 1e-6))
            )
        return np.stack(costs, axis=2).astype(np.float32), np.sqrt(
            np.maximum(left_var, 0)
        ).astype(np.float32)

    def _select(
        self, costs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        best_index = np.argmax(costs, axis=2)
        best = np.take_along_axis(costs, best_index[..., None], axis=2)[..., 0]
        disparity = best_index.astype(np.float32) + self.min_disparity
        if costs.shape[2] > 2:
            rows, columns = np.indices(best_index.shape)
            interior = (best_index > 0) & (best_index < costs.shape[2] - 1)
            previous = costs[rows, columns, np.maximum(best_index - 1, 0)]
            following = costs[
                rows, columns, np.minimum(best_index + 1, costs.shape[2] - 1)
            ]
            denominator = previous - 2.0 * best + following
            delta = np.zeros_like(disparity)
            stable = interior & (np.abs(denominator) > 1e-6)
            delta[stable] = np.clip(
                0.5 * (previous[stable] - following[stable]) / denominator[stable],
                -0.5,
                0.5,
            )
            disparity += delta
        runner = np.full(best.shape, -1.0, dtype=np.float32)
        for index in range(costs.shape[2]):
            eligible = np.abs(index - best_index) > 1
            runner = np.where(eligible, np.maximum(runner, costs[..., index]), runner)
        return disparity, best, runner

    def compute(
        self, rectified: RectifiedStereoPair
    ) -> DisparityResult:
        forward_cost, left_std = self._cost_volume(
            rectified.left_ultrawide, rectified.right_main, False
        )
        forward, best, runner = self._select(forward_cost)
        del forward_cost
        reverse_cost, right_std = self._cost_volume(
            rectified.right_main, rectified.left_ultrawide, True
        )
        reverse, _, _ = self._select(reverse_cost)
        del reverse_cost

        height, width = forward.shape
        yy, xx = np.indices((height, width))
        matched_x = np.rint(xx - forward).astype(np.int32)
        in_bounds = (matched_x >= 0) & (matched_x < width)
        safe_x = np.clip(matched_x, 0, width - 1)
        reverse_at_match = reverse[yy, safe_x]
        residual = np.abs(forward - reverse_at_match)
        matched_right_std = right_std[yy, safe_x]
        texture_std = np.minimum(left_std, matched_right_std)

        left_right_agreement = np.exp(-np.square(np.minimum(residual, 10.0)))
        best_zncc = np.clip((best - 0.25) / 0.75, 0.0, 1.0)
        distinctiveness = np.clip((best - runner) / 0.25, 0.0, 1.0)
        texture_support = np.clip(texture_std / 20.0, 0.0, 1.0)
        confidence = np.power(
            left_right_agreement
            * best_zncc
            * distinctiveness
            * texture_support,
            0.25,
        ).astype(np.float32)

        radius = self.window_size // 2
        validity = (
            rectified.joint_validity
            & np.isfinite(forward)
            & (forward > self.min_disparity)
            & (forward < self.max_disparity - 1)
            & in_bounds
            & (residual <= 1.0)
            & (best_zncc >= 0.10)
            & (texture_support >= 0.15)
        )
        validity[:radius, :] = False
        validity[-radius:, :] = False
        validity[:, : self.max_disparity + radius] = False
        validity[:, -radius:] = False
        confidence = np.where(validity, confidence, 0.0).astype(np.float32)
        return DisparityResult(
            left_to_right_px=forward.astype(np.float32),
            right_to_left_px=reverse.astype(np.float32),
            confidence=confidence,
            validity=validity,
            evidence={
                "left_right_agreement": left_right_agreement.astype(np.float32),
                "best_zncc": best_zncc.astype(np.float32),
                "distinctiveness": distinctiveness.astype(np.float32),
                "texture_support": texture_support.astype(np.float32),
            },
            method=MatcherMethod.SLIDING_WINDOW,
            profile_version=self.PROFILE_VERSION,
        )


class IPhoneStereoProcessor:
    """One public operation for rectification, evidence matching, and metric depth."""

    def __init__(
        self,
        calibration_result: CalibrationResult,
        output_size: PixelSize | None = None,
    ) -> None:
        self.calibration_result = calibration_result
        if calibration_result.status is CalibrationStatus.REJECTED:
            self.output_size = output_size
            self.rectifier = None
        else:
            self.output_size = output_size or (
                calibration_result.processing_geometry.main_native_size.fit_within_width(640)
            )
            self.rectifier = IPhoneStereoRectifier(
                calibration_result, self.output_size
            )
        self.sliding_window = EvidenceSlidingWindowMatcher()

    def process(
        self,
        main: np.ndarray,
        ultrawide: np.ndarray,
        orientations: SourceOrientations | None = None,
        method: MatcherMethod = MatcherMethod.SLIDING_WINDOW,
    ) -> ProcessedStereoResult:
        if self.calibration_result.status is CalibrationStatus.REJECTED:
            return ProcessedStereoResult(
                disposition=ProcessingDisposition.REJECTED,
                rectified=None,
                disparity=None,
                depth=None,
                calibration=self.calibration_result,
                trusted_depth_eligible=False,
                reason_codes=(
                    self.calibration_result.reason_code or "CALIBRATION_REJECTED",
                ),
                timings_ms={},
            )

        assert self.rectifier is not None
        if orientations is None:
            orientations = SourceOrientations(
                main_to_calibration_native=_orientation_to_native(
                    main,
                    self.calibration_result.processing_geometry.main_native_size,
                    "Main",
                ),
                ultrawide_to_calibration_native=_orientation_to_native(
                    ultrawide,
                    self.calibration_result.processing_geometry.ultrawide_native_size,
                    "Ultra-Wide",
                ),
            )
        timings: dict[str, float] = {}
        start = time.perf_counter()
        rectified = self.rectifier.rectify(main, ultrawide, orientations)
        timings["rectification"] = (time.perf_counter() - start) * 1000
        if rectified.diagnostics["joint_valid_fraction"] < 0.5:
            raise ContractError("Rectification joint validity is below 50 percent")

        start = time.perf_counter()
        if method is not MatcherMethod.SLIDING_WINDOW:
            raise ContractError("Census-SGBM diagnostic method is not implemented yet")
        disparity = self.sliding_window.compute(rectified)
        timings["matching_and_confidence"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        geometry = rectified.geometry
        denominator = disparity.left_to_right_px + geometry.disparity_offset_px
        depth = np.full(disparity.left_to_right_px.shape, np.nan, dtype=np.float32)
        valid = disparity.validity & np.isfinite(denominator) & (denominator > 1e-3)
        depth[valid] = (
            geometry.focal_length_px * geometry.baseline_m / denominator[valid]
        )
        valid &= np.isfinite(depth) & (depth >= 0.1) & (depth <= 15.0)
        depth[~valid] = np.nan
        metric_depth = MetricDepthResult(meters=depth, validity=valid)
        timings["metric_depth"] = (time.perf_counter() - start) * 1000

        reason_codes = [
            value
            for value in (self.calibration_result.reason_code,)
            if value is not None
        ]
        disparity_coverage = float(np.mean(disparity.validity))
        depth_coverage = float(np.mean(metric_depth.validity))
        if disparity_coverage < 0.10:
            reason_codes.append("DISPARITY_VALIDITY_BELOW_THRESHOLD")
        if depth_coverage < 0.05:
            reason_codes.append("DEPTH_VALIDITY_BELOW_THRESHOLD")

        calibration_trusted = (
            self.calibration_result.status is CalibrationStatus.TRUSTED
        )
        trusted_depth_eligible = calibration_trusted and not reason_codes
        disposition = (
            ProcessingDisposition.TRUSTED
            if trusted_depth_eligible
            else ProcessingDisposition.DIAGNOSTIC
        )
        return ProcessedStereoResult(
            disposition=disposition,
            rectified=rectified,
            disparity=disparity,
            depth=metric_depth,
            calibration=self.calibration_result,
            trusted_depth_eligible=trusted_depth_eligible,
            reason_codes=tuple(reason_codes),
            timings_ms=timings,
        )
