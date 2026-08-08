"""
Heterogeneous Stereo Rectification Engine for iPhone Dual-Camera Pairs (Main Wide + Ultra-Wide).
Handles FOV alignment, dynamic calibration scaling, and single-pass cv2.stereoRectify remapping.
"""

import math
import cv2
import numpy as np


class HeterogeneousStereoRectifier:
    """
    Handles FOV alignment, dynamic epipolar rectification, and disparity-to-depth
    conversion for heterogeneous iPhone stereo pairs (Main + Ultra-Wide).
    """

    def __init__(self, target_size=(1280, 960)):
        """
        :param target_size: (width, height) tuple for working resolution.
        """
        self.target_size = target_size

    def scale_intrinsics(self, K: np.ndarray, orig_size: tuple, target_size: tuple) -> np.ndarray:
        """
        Scales 3x3 intrinsics matrix K linearly from reference dimensions to target working resolution.
        """
        sx = target_size[0] / float(orig_size[0])
        sy = target_size[1] / float(orig_size[1])
        K_scaled = K.astype(np.float64).copy()
        K_scaled[0, 0] *= sx  # fx
        K_scaled[1, 1] *= sy  # fy
        K_scaled[0, 2] *= sx  # cx
        K_scaled[1, 2] *= sy  # cy
        return K_scaled

    def rectify_pair(self, img_main: np.ndarray, img_uw: np.ndarray,
                     K1: np.ndarray, D1: np.ndarray,
                     K2: np.ndarray, D2: np.ndarray,
                     R: np.ndarray, T: np.ndarray):
        """
        Computes dynamic rectification and maps images into coplanar, horizontally aligned epipolar space.

        :param img_main: Main camera RGB array (H1, W1, 3)
        :param img_uw: Ultra-Wide camera RGB array (H2, W2, 3)
        :param K1: 3x3 Main camera intrinsics
        :param D1: Distortion coeffs for Main camera (or zeros)
        :param K2: 3x3 Ultra-Wide intrinsics
        :param D2: Distortion coeffs for Ultra-Wide (or zeros)
        :param R: 3x3 Rotation matrix (Ultra-Wide to Main)
        :param T: 3x1 Translation vector (Ultra-Wide to Main) in mm or meters
        :return: (rectified_main, rectified_uw, P1, P2, Q)
        """
        size1 = (img_main.shape[1], img_main.shape[0])
        size2 = (img_uw.shape[1], img_uw.shape[0])

        # 1. Scale intrinsics to working resolution
        K1_scaled = self.scale_intrinsics(K1, size1, self.target_size)
        K2_scaled = self.scale_intrinsics(K2, size2, self.target_size)

        D1_flat = D1.flatten() if D1 is not None else np.zeros(5, dtype=np.float64)
        D2_flat = D2.flatten() if D2 is not None else np.zeros(5, dtype=np.float64)

        R_mat = R.astype(np.float64)
        T_vec = T.flatten().reshape(3, 1).astype(np.float64)

        # 2. Compute dynamic rectification matrices
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=K1_scaled, distCoeffs1=D1_flat,
            cameraMatrix2=K2_scaled, distCoeffs2=D2_flat,
            imageSize=self.target_size,
            R=R_mat, T=T_vec,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=self.target_size
        )

        # 3. Build single-pass coordinate lookup tables
        map1_x, map1_y = cv2.initUndistortRectifyMap(K1_scaled, D1_flat, R1, P1, self.target_size, cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(K2_scaled, D2_flat, R2, P2, self.target_size, cv2.CV_32FC1)

        # 4. Execute single-pass remapping
        rectified_main = cv2.remap(img_main, map1_x, map1_y, cv2.INTER_LINEAR)
        rectified_uw = cv2.remap(img_uw, map2_x, map2_y, cv2.INTER_LINEAR)

        return rectified_main, rectified_uw, P1, P2, Q

    @staticmethod
    def disparity_to_depth(disparity: np.ndarray, f_rect: float, baseline: float, doffs: float = 0.0) -> np.ndarray:
        """
        Calculates metric depth Z = (f_rect * B) / (disparity + doffs).

        :param disparity: 2D array of pixel disparities.
        :param f_rect: Rectified focal length in pixels.
        :param baseline: Physical baseline B in meters.
        :param doffs: Disparity offset in pixels.
        :return: 2D array of metric depth in meters.
        """
        valid_mask = disparity > 0
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[valid_mask] = (f_rect * baseline) / (disparity[valid_mask] + doffs)
        return depth
