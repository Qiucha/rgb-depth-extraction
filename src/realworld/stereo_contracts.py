"""Named contracts for limitation-aware iPhone stereo processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class ContractError(ValueError):
    """Raised before numerical processing when an interface invariant is broken."""


class CalibrationStatus(str, Enum):
    TRUSTED = "trusted"
    INPUT_LIMITED = "input_limited"
    REJECTED = "rejected"


class ProcessingDisposition(str, Enum):
    TRUSTED = "trusted"
    DIAGNOSTIC = "diagnostic"
    REJECTED = "rejected"


class MatcherMethod(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    CENSUS_SGBM = "census_sgbm"


class QuarterTurn(str, Enum):
    IDENTITY = "identity"
    CLOCKWISE_90 = "clockwise_90"
    ROTATE_180 = "rotate_180"
    COUNTERCLOCKWISE_90 = "counterclockwise_90"


@dataclass(frozen=True)
class PixelSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ContractError("Pixel dimensions must be positive")

    @property
    def cv(self) -> tuple[int, int]:
        return self.width, self.height

    def fit_within_width(self, maximum_width: int) -> "PixelSize":
        if maximum_width <= 0:
            raise ContractError("Maximum pixel width must be positive")
        if self.width <= maximum_width:
            return self
        scale = maximum_width / self.width
        return PixelSize(maximum_width, max(1, round(self.height * scale)))


@dataclass(frozen=True)
class SourceOrientations:
    main_to_calibration_native: QuarterTurn = QuarterTurn.IDENTITY
    ultrawide_to_calibration_native: QuarterTurn = QuarterTurn.IDENTITY


@dataclass(frozen=True)
class CalibrationGeometry:
    main_matrix: np.ndarray
    main_distortion: np.ndarray
    ultrawide_matrix: np.ndarray
    ultrawide_distortion: np.ndarray
    main_to_ultrawide_rotation: np.ndarray
    main_to_ultrawide_translation_m: np.ndarray
    main_native_size: PixelSize
    ultrawide_native_size: PixelSize
    lens_model: str = "rational"

    def __post_init__(self) -> None:
        arrays = {
            "main_matrix": (self.main_matrix, (3, 3)),
            "ultrawide_matrix": (self.ultrawide_matrix, (3, 3)),
            "main_to_ultrawide_rotation": (
                self.main_to_ultrawide_rotation,
                (3, 3),
            ),
        }
        for name, (array, shape) in arrays.items():
            if np.asarray(array).shape != shape or not np.isfinite(array).all():
                raise ContractError(f"{name} must be a finite {shape} matrix")
        translation = np.asarray(self.main_to_ultrawide_translation_m).reshape(-1)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ContractError("Main-to-Ultra-Wide translation must be a finite 3-vector")
        baseline = float(np.linalg.norm(translation))
        if not 0.005 <= baseline <= 0.1:
            raise ContractError("Calibration baseline must be explicitly expressed in meters")
        if abs(float(np.linalg.det(self.main_to_ultrawide_rotation)) - 1.0) > 1e-5:
            raise ContractError("Calibration rotation must be proper")
        for matrix in (self.main_matrix, self.ultrawide_matrix):
            if min(float(matrix[0, 0]), float(matrix[1, 1])) <= 0:
                raise ContractError("Calibration focal lengths must be positive")


@dataclass(frozen=True)
class CalibrationResult:
    status: CalibrationStatus
    reason_code: str | None
    recommended_action: str | None
    trusted_calibration: CalibrationGeometry | None
    diagnostic_candidate: CalibrationGeometry | None
    source_path: str
    fingerprint: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is CalibrationStatus.TRUSTED:
            if self.trusted_calibration is None or self.diagnostic_candidate is not None:
                raise ContractError("Trusted result requires only trusted_calibration")
        elif self.status is CalibrationStatus.INPUT_LIMITED:
            if self.trusted_calibration is not None or not self.reason_code:
                raise ContractError("Input-limited result cannot contain trusted calibration")
        elif self.trusted_calibration is not None or self.diagnostic_candidate is not None:
            raise ContractError("Rejected result cannot contain calibration geometry")

    @property
    def processing_geometry(self) -> CalibrationGeometry:
        geometry = self.trusted_calibration or self.diagnostic_candidate
        if geometry is None:
            raise ContractError(
                "Calibration Result has no trusted calibration or Diagnostic Calibration Candidate"
            )
        return geometry


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _size(value: Any, name: str) -> PixelSize:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ContractError(f"{name} must be [width, height]")
    return PixelSize(int(value[0]), int(value[1]))


def _geometry(data: dict[str, Any]) -> CalibrationGeometry:
    return CalibrationGeometry(
        main_matrix=_array(data["K_main"]),
        main_distortion=_array(data["D_main"]).reshape(-1),
        ultrawide_matrix=_array(data["K_ultrawide"]),
        ultrawide_distortion=_array(data["D_ultrawide"]).reshape(-1),
        main_to_ultrawide_rotation=_array(data["R_main_to_ultrawide"]),
        main_to_ultrawide_translation_m=_array(
            data["T_main_to_ultrawide_m"]
        ).reshape(3),
        main_native_size=_size(data["main_native_size"], "main_native_size"),
        ultrawide_native_size=_size(
            data["ultrawide_native_size"], "ultrawide_native_size"
        ),
        lens_model=str(data.get("lens_model", "rational")),
    )


def load_calibration_result(path: str | Path) -> CalibrationResult:
    source = Path(path)
    raw_bytes = source.read_bytes()
    data = json.loads(raw_bytes)
    try:
        status = CalibrationStatus(data["status"])
    except (KeyError, ValueError) as error:
        raise ContractError("Unknown or missing Calibration Result status") from error
    trusted = data.get("trusted_calibration")
    candidate = data.get("diagnostic_candidate")
    return CalibrationResult(
        status=status,
        reason_code=data.get("reason_code"),
        recommended_action=data.get("recommended_action"),
        trusted_calibration=_geometry(trusted) if trusted else None,
        diagnostic_candidate=_geometry(candidate) if candidate else None,
        source_path=str(source),
        fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
        evidence={
            key: value
            for key, value in data.items()
            if key not in {"trusted_calibration", "diagnostic_candidate"}
        },
    )


@dataclass(frozen=True)
class RectificationGeometry:
    left_projection: np.ndarray
    right_projection: np.ndarray
    q_matrix: np.ndarray
    focal_length_px: float
    baseline_m: float
    disparity_offset_px: float


@dataclass(frozen=True)
class RectifiedStereoPair:
    left_ultrawide: np.ndarray
    right_main: np.ndarray
    left_validity: np.ndarray
    right_validity: np.ndarray
    joint_validity: np.ndarray
    geometry: RectificationGeometry
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DisparityResult:
    left_to_right_px: np.ndarray
    right_to_left_px: np.ndarray
    confidence: np.ndarray
    validity: np.ndarray
    evidence: dict[str, np.ndarray]
    method: MatcherMethod
    profile_version: str


@dataclass(frozen=True)
class MetricDepthResult:
    meters: np.ndarray
    validity: np.ndarray


@dataclass(frozen=True)
class ProcessedStereoResult:
    disposition: ProcessingDisposition
    rectified: RectifiedStereoPair | None
    disparity: DisparityResult | None
    depth: MetricDepthResult | None
    calibration: CalibrationResult
    trusted_depth_eligible: bool
    reason_codes: tuple[str, ...]
    timings_ms: dict[str, float]

    def __post_init__(self) -> None:
        artifacts = (self.rectified, self.disparity, self.depth)
        if self.disposition is ProcessingDisposition.REJECTED:
            if any(artifact is not None for artifact in artifacts):
                raise ContractError("Rejected processing cannot expose numerical artifacts")
            if self.trusted_depth_eligible:
                raise ContractError("Rejected processing cannot be trusted")
        elif any(artifact is None for artifact in artifacts):
            raise ContractError("Processed results require all numerical artifacts")
