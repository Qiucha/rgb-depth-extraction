"""
Real-world iPhone Dual-Camera & Intel RealSense Stereo Depth Evaluation Package.
"""

from .hetero_rectifier import HeterogeneousStereoRectifier
from .iphone_stereo_processor import IPhoneStereoProcessor, IPhoneStereoRectifier
from .stereo_contracts import (
    CalibrationResult,
    MatcherMethod,
    PixelSize,
    ProcessedStereoResult,
    ProcessingDisposition,
    load_calibration_result,
)
from .dataset_loader import RealWorldDatasetLoader
from .realsense_icp import RealSensePointcloudAligner
from .evaluator import RealWorldEvaluator
from .sync_focus_validator import SensorSyncFocusValidator
from .psf_gradient_optimizer import PSFGradientOptimizer
from .pipeline import run_realworld_pipeline

__all__ = [
    'HeterogeneousStereoRectifier',
    'IPhoneStereoProcessor',
    'IPhoneStereoRectifier',
    'CalibrationResult',
    'MatcherMethod',
    'PixelSize',
    'ProcessedStereoResult',
    'ProcessingDisposition',
    'load_calibration_result',
    'RealWorldDatasetLoader',
    'RealSensePointcloudAligner',
    'RealWorldEvaluator',
    'SensorSyncFocusValidator',
    'PSFGradientOptimizer',
    'run_realworld_pipeline'
]
