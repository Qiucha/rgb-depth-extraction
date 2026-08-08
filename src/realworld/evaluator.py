"""
RealWorld Stereo Evaluation Suite comparing iPhone passive stereo depth against Intel RealSense benchmark.
Computes MAE, RMSE, Texture Dependency Error Ratio, and Boundary Flying Pixel Ratios.
"""

import cv2
import numpy as np


class RealWorldEvaluator:
    """
    Evaluates iPhone depth map against RealSense active IR ground-truth depth map across 3 primary axes.
    """

    def evaluate_all(self, depth_est: np.ndarray, depth_gt: np.ndarray, rgb_img: np.ndarray = None):
        """
        Runs comprehensive evaluation suite.

        :param depth_est: Estimated iPhone metric depth map (H, W) in meters.
        :param depth_gt: Ground truth RealSense metric depth map (H, W) in meters.
        :param rgb_img: Corresponding RGB image (H, W, 3) for texture and edge analysis.
        :return: Dict containing MAE, RMSE, bad_pixel_ratio, texture_ratio, flying_pixels_ratio.
        """
        valid_mask = (depth_gt > 0.1) & (depth_gt < 10.0) & (depth_est > 0.1) & (depth_est < 10.0)
        
        if not np.any(valid_mask):
            return {
                "mae_m": 0.0,
                "rmse_m": 0.0,
                "bad_pixel_ratio": 0.0,
                "texture_dependency_ratio": 1.0,
                "flying_pixel_ratio": 0.0
            }

        diff = depth_est[valid_mask] - depth_gt[valid_mask]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bad_pixels = float(np.mean(np.abs(diff) > 0.05))  # > 5 cm error

        texture_ratio = self.evaluate_texture_dependency(depth_est, depth_gt, rgb_img)
        flying_ratio = self.evaluate_flying_pixels(depth_est, rgb_img)

        return {
            "mae_m": mae,
            "rmse_m": rmse,
            "bad_pixel_ratio": bad_pixels,
            "texture_dependency_ratio": texture_ratio,
            "flying_pixel_ratio": flying_ratio
        }

    def evaluate_texture_dependency(self, depth_est: np.ndarray, depth_gt: np.ndarray, rgb_img: np.ndarray = None) -> float:
        """
        Computes the ratio of MAE error on textureless surfaces vs textured surfaces.
        High ratio (> 2.0) indicates strong texture dependency failure (typical for passive stereo without IR).
        """
        if rgb_img is None:
            return 1.0

        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY) if len(rgb_img.shape) == 3 else rgb_img
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        valid_mask = (depth_gt > 0.1) & (depth_gt < 10.0) & (depth_est > 0.1)
        textured_mask = valid_mask & (grad_mag > 20.0)
        textureless_mask = valid_mask & (grad_mag <= 20.0)

        err_textured = np.mean(np.abs(depth_est[textured_mask] - depth_gt[textured_mask])) if np.any(textured_mask) else 0.001
        err_textureless = np.mean(np.abs(depth_est[textureless_mask] - depth_gt[textureless_mask])) if np.any(textureless_mask) else 0.001

        return float(err_textureless / (err_textured + 1e-6))

    def evaluate_flying_pixels(self, depth_est: np.ndarray, rgb_img: np.ndarray = None) -> float:
        """
        Measures depth gradient discontinuities near RGB object edges (flying pixels / depth bleed).
        """
        if rgb_img is None:
            return 0.0

        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY) if len(rgb_img.shape) == 3 else rgb_img
        edges = cv2.Canny(gray, 50, 150) > 0

        depth_grad_x = cv2.Sobel(depth_est, cv2.CV_64F, 1, 0, ksize=3)
        depth_grad_y = cv2.Sobel(depth_est, cv2.CV_64F, 0, 1, ksize=3)
        depth_grad_mag = np.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)

        # Flying pixels occur where high depth gradients extend beyond high RGB edges
        flying_mask = (depth_grad_mag > 0.2) & (~edges)
        return float(np.mean(flying_mask))
