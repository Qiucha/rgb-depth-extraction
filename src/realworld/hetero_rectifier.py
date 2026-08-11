"""
Heterogeneous Stereo Rectification Engine for iPhone Dual-Camera Pairs (Main Wide + Ultra-Wide).
Handles FOV alignment, dynamic calibration scaling, fisheye lens distortion models,
and strict cy principal point optical center alignment.
"""

import math
import cv2
import numpy as np


class HeterogeneousStereoRectifier:
    """
    Handles FOV alignment, dynamic epipolar rectification, fisheye distortion modeling,
    and disparity-to-depth conversion for heterogeneous stereo pairs (Main + Ultra-Wide).
    """

    def __init__(self, target_size=(1280, 960), is_fisheye: bool = False):
        """
        :param target_size: (width, height) tuple for working resolution.
        :param is_fisheye: If True, uses cv2.fisheye stereo rectification model.
        """
        self.target_size = target_size
        self.is_fisheye = is_fisheye

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

    def _rotate_intrinsics_ccw90(self, K: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """
        Transforms 3x3 intrinsic matrix K for a 90-degree counter-clockwise image rotation.
        New pixel coordinates (x', y') = (y, orig_w - 1 - x).
        """
        K_rot = K.astype(np.float64).copy()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        K_rot[0, 0] = fy
        K_rot[1, 1] = fx
        K_rot[0, 2] = cy
        K_rot[1, 2] = orig_w - cx
        return K_rot

    def rectify_pair(self, img_main: np.ndarray, img_uw: np.ndarray,
                     K1: np.ndarray, D1: np.ndarray,
                     K2: np.ndarray, D2: np.ndarray,
                     R: np.ndarray, T: np.ndarray,
                     is_fisheye: bool = None):
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
        :param is_fisheye: Optional override for fisheye model enabling
        :return: (rectified_main, rectified_uw, P1, P2, Q)
        """
        use_fisheye = is_fisheye if is_fisheye is not None else self.is_fisheye

        # 0. Auto-rotate portrait images (H > W) to landscape to align physical baseline along X
        if img_main.shape[0] > img_main.shape[1]:
            orig_h1, orig_w1 = img_main.shape[0], img_main.shape[1]
            orig_h2, orig_w2 = img_uw.shape[0], img_uw.shape[1]
            img_main = cv2.rotate(img_main, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_uw = cv2.rotate(img_uw, cv2.ROTATE_90_COUNTERCLOCKWISE)
            K1 = self._rotate_intrinsics_ccw90(K1, orig_w1, orig_h1)
            K2 = self._rotate_intrinsics_ccw90(K2, orig_w2, orig_h2)
            # Transform extrinsics to match the rotated coordinate system
            R_z90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
            R = R_z90 @ R.astype(np.float64) @ R_z90.T
            T = R_z90 @ T.flatten().reshape(3, 1).astype(np.float64)

        size1 = (img_main.shape[1], img_main.shape[0])
        size2 = (img_uw.shape[1], img_uw.shape[0])

        # 1. Match working target_size aspect ratio to input image aspect ratio if not explicitly specified
        if self.target_size is None:
            aspect_ratio = size1[0] / float(size1[1])
            if abs(aspect_ratio - (16.0 / 9.0)) < 0.1:
                target_size = (1280, 720)
            elif abs(aspect_ratio - (4.0 / 3.0)) < 0.1:
                target_size = (1280, 960)
            else:
                target_size = (1280, int(round(1280 / aspect_ratio)))
            self.target_size = target_size
        else:
            target_size = self.target_size

        # Scale intrinsics to working resolution
        K1_scaled = self.scale_intrinsics(K1, size1, self.target_size)
        K2_scaled = self.scale_intrinsics(K2, size2, self.target_size)

        R_mat = R.astype(np.float64)
        T_vec = T.flatten().reshape(3, 1).astype(np.float64)

        if use_fisheye:
            # Fisheye model expects 4 distortion coefficients (k1, k2, k3, k4)
            D1_4 = D1.flatten()[:4] if D1 is not None and len(D1.flatten()) >= 4 else np.zeros((4, 1), dtype=np.float64)
            D2_4 = D2.flatten()[:4] if D2 is not None and len(D2.flatten()) >= 4 else np.zeros((4, 1), dtype=np.float64)
            
            D1_4 = D1_4.reshape(4, 1)
            D2_4 = D2_4.reshape(4, 1)

            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
                K1=K1_scaled, D1=D1_4,
                K2=K2_scaled, D2=D2_4,
                imageSize=self.target_size,
                R=R_mat, tvec=T_vec,
                flags=cv2.CALIB_ZERO_DISPARITY,
                newImageSize=self.target_size
            )

            map1_x, map1_y = cv2.fisheye.initUndistortRectifyMap(K1_scaled, D1_4, R1, P1, self.target_size, cv2.CV_32FC1)
            map2_x, map2_y = cv2.fisheye.initUndistortRectifyMap(K2_scaled, D2_4, R2, P2, self.target_size, cv2.CV_32FC1)
        else:
            D1_flat = D1.flatten() if D1 is not None else np.zeros(5, dtype=np.float64)
            D2_flat = D2.flatten() if D2 is not None else np.zeros(5, dtype=np.float64)

            # 2. Compute dynamic standard pinhole rectification matrices
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

        # Derive Q matrix from actual P1/P2 values computed by stereoRectify
        f_rect = float(P1[0, 0])
        cx1 = float(P1[0, 2])
        cy1 = float(P1[1, 2])
        # Baseline from P2[0,3] = -f * Tx, so Tx = -P2[0,3] / f
        baseline_from_P2 = abs(float(P2[0, 3])) / max(f_rect, 1e-6)
        # doffs = cx_right - cx_left (disparity offset from different principal points)
        doffs = float(P2[0, 2]) - cx1
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = 1.0
        Q[0, 3] = -cx1
        Q[1, 1] = 1.0
        Q[1, 3] = -cy1
        Q[2, 3] = f_rect
        Q[3, 2] = -1.0 / max(baseline_from_P2, 1e-6)
        Q[3, 3] = doffs / max(baseline_from_P2, 1e-6)


        # 4. Execute single-pass remapping with Lanczos anti-aliased interpolation
        rectified_main = cv2.remap(img_main, map1_x, map1_y, cv2.INTER_LANCZOS4)
        rectified_uw = cv2.remap(img_uw, map2_x, map2_y, cv2.INTER_LANCZOS4)


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
