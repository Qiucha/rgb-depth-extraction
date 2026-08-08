"""
Deep Disparity Model Integration Adapter Interface.
Provides unified abstraction for classical sliding window block matching on stereo pairs
and deep stereo neural networks (CREStereo, RAFT-Stereo, AnyStereo).
"""

from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
from typing import Tuple, Optional, Any, Dict


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Helper to convert BGR or RGB image matrix to single-channel grayscale."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Helper to convert single-channel or BGR image matrix to 3-channel RGB."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class BaseStereoMatcher(ABC):
    """
    Abstract base class for stereo pair disparity matching backends.
    """

    @abstractmethod
    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute disparity map from left and right rectified stereo pair images.

        :param img_left: Left rectified image (H, W) or (H, W, 3)
        :param img_right: Right rectified image (H, W) or (H, W, 3)
        :return: Tuple of (disparity_map [float32], validity_mask [bool])
        """
        pass


class ClassicalSlidingWindowAdapter(BaseStereoMatcher):
    """
    Adapter wrapping the 1D Epipolar Sliding Window block matcher (NCC / ZNCC / SAD / SSD).
    """

    def __init__(self, window_size: int = 7, max_disparity: int = 64, metric: str = "zncc"):
        from src.stereo_depth.sliding_window import SlidingWindowMatcher
        self.metric = metric.lower()
        self.matcher = SlidingWindowMatcher(window_size=window_size, max_disparity=max_disparity, metric=self.metric)

    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray_left = ensure_grayscale(img_left)
        gray_right = ensure_grayscale(img_right)
        return self.matcher.compute_disparity(gray_left, gray_right)


class DeepDisparityMatcherAdapter(BaseStereoMatcher):
    """
    Adapter for deep learning stereo matching architectures (CREStereo, RAFT-Stereo, AnyStereo).
    Supports ONNX Runtime (.onnx) and PyTorch (.pt, .pth) backends with graceful fallback to classical block matching.
    """

    def __init__(self, model_type: str = "cre_stereo", model_path: Optional[str] = None, max_disparity: int = 128, **kwargs):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.max_disparity = max_disparity
        self.kwargs = kwargs

        self.session: Optional[Any] = None
        self.torch_model: Optional[Any] = None
        self.backend: str = "none"

        self.fallback_matcher = ClassicalSlidingWindowAdapter(max_disparity=min(max_disparity, 64))

        if self.model_path and os.path.exists(self.model_path):
            self._initialize_model()

    def _initialize_model(self):
        """Attempts loading ONNX Runtime or PyTorch model backend depending on extension."""
        if not self.model_path:
            return

        if self.model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                self.backend = "onnx"
            except ImportError:
                print(f"[DeepDisparityMatcherAdapter] onnxruntime module not installed. Falling back to classical matcher.")
            except Exception as e:
                print(f"[DeepDisparityMatcherAdapter] Failed to load ONNX model at {self.model_path}: {e}")

        elif self.model_path.endswith((".pth", ".pt")):
            try:
                import torch
                self.torch_model = torch.jit.load(self.model_path) if self.model_path.endswith(".pt") else torch.load(self.model_path)
                if hasattr(self.torch_model, "eval"):
                    self.torch_model.eval()
                self.backend = "torch"
            except ImportError:
                print(f"[DeepDisparityMatcherAdapter] PyTorch module not installed. Falling back to classical matcher.")
            except Exception as e:
                print(f"[DeepDisparityMatcherAdapter] Failed to load PyTorch model at {self.model_path}: {e}")

    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes deep network inference or falls back to classical block matching.
        """
        if self.backend == "onnx" and self.session is not None:
            try:
                return self._run_onnx_inference(img_left, img_right)
            except Exception as e:
                print(f"[DeepDisparityMatcherAdapter] ONNX Inference error: {e}. Using fallback matcher.")

        elif self.backend == "torch" and self.torch_model is not None:
            try:
                return self._run_torch_inference(img_left, img_right)
            except Exception as e:
                print(f"[DeepDisparityMatcherAdapter] Torch Inference error: {e}. Using fallback matcher.")

        # Fallback when model is missing or unsupported
        return self.fallback_matcher.compute_disparity(img_left, img_right)

    def _format_inputs(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rgb_left = ensure_rgb(img_left)
        rgb_right = ensure_rgb(img_right)
        tensor_left = (rgb_left.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, :]
        tensor_right = (rgb_right.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, :]
        return tensor_left, tensor_right

    def _process_model_output(self, raw_output: Any, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Processes raw model output array/tuple considering architecture conventions (e.g. RAFT-Stereo list vs CREStereo single map)."""
        h, w = target_shape
        disparity = raw_output

        # RAFT-Stereo returns a list/sequence of predictions per iteration
        if isinstance(disparity, (list, tuple)):
            disparity = disparity[-1]

        if hasattr(disparity, "detach"):
            disparity = disparity.detach().cpu().numpy()

        if disparity.ndim == 4:
            disparity = disparity[0, 0]
        elif disparity.ndim == 3:
            disparity = disparity[0]

        if disparity.shape != (h, w):
            disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)

        valid_mask = (disparity > 0.0) & (disparity < self.max_disparity)
        return disparity.astype(np.float32), valid_mask

    def _run_onnx_inference(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img_left.shape[:2]
        t_left, t_right = self._format_inputs(img_left, img_right)
        input_names = [i.name for i in self.session.get_inputs()]
        outputs = self.session.run(None, {input_names[0]: t_left, input_names[1]: t_right})
        return self._process_model_output(outputs[0], (h, w))

    def _run_torch_inference(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        h, w = img_left.shape[:2]
        t_left, t_right = self._format_inputs(img_left, img_right)
        torch_left = torch.from_numpy(t_left)
        torch_right = torch.from_numpy(t_right)
        with torch.no_grad():
            outputs = self.torch_model(torch_left, torch_right)
        return self._process_model_output(outputs, (h, w))


def get_stereo_matcher(matcher_name: str = "sliding_window", **kwargs) -> BaseStereoMatcher:
    """
    Factory function for constructing stereo pair matcher instances.

    :param matcher_name: Name of matcher ('sliding_window', 'zncc', 'ncc', 'sad', 'ssd', 'cre_stereo', 'raft_stereo', 'anystereo')
    :param kwargs: Additional arguments for matcher initialization
    :return: Instance of BaseStereoMatcher
    """
    name = matcher_name.lower()
    if name in ["sliding_window", "zncc", "ncc", "sad", "ssd"]:
        metric = name if name in ["zncc", "ncc", "sad", "ssd"] else kwargs.pop("metric", "zncc")
        return ClassicalSlidingWindowAdapter(metric=metric, **kwargs)
    elif name in ["cre_stereo", "raft_stereo", "anystereo", "deep"]:
        return DeepDisparityMatcherAdapter(model_type=name, **kwargs)
    else:
        raise ValueError(f"Unknown stereo matcher name: '{matcher_name}'. Supported options: 'sliding_window', 'zncc', 'ncc', 'sad', 'ssd', 'cre_stereo', 'raft_stereo', 'anystereo'")
