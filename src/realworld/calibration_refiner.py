"""
ChArUco & Checkerboard Calibration Refinement Tool for Main + Ultra-Wide Stereo Pairs.
Detects calibration pattern corners and refines camera intrinsics (K1, K2), distortion (D1, D2),
and extrinsic baseline pose (R, T) with automatic grid discovery and iterative outlier pruning.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class CalibrationRefiner:
    """
    Checkerboard & ChArUco target corner detector and stereo calibration parameter solver.
    Includes pattern auto-discovery, sector-based/sub-pixel corner extraction,
    and iterative outlier pruning.
    """

    def __init__(
        self,
        pattern_size: Tuple[int, int] = (9, 6),
        square_size_m: float = 0.025,
        auto_discover: bool = False
    ):
        """
        :param pattern_size: (cols, rows) inner corner grid dimensions.
        :param square_size_m: Physical checkerboard square size in meters.
        :param auto_discover: Enable searching candidate grid patterns if specified pattern fails.
        """
        self.pattern_size = pattern_size
        self.square_size_m = square_size_m
        self.auto_discover = auto_discover

        cols, rows = pattern_size
        self.objp = np.zeros((cols * rows, 3), dtype=np.float32)
        self.objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

    def auto_discover_pattern_size(self, image_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[int, int]:
        """
        Scans a list of image pairs using fast check candidate pattern grid sizes
        to automatically discover the inner checkerboard corner dimensions.
        """
        candidate_patterns = [
            self.pattern_size,
            (self.pattern_size[1], self.pattern_size[0]),
            (5, 5), (5, 4), (4, 5), (9, 6), (6, 9), (8, 6), (6, 8),
            (7, 5), (5, 7), (10, 7), (7, 10), (6, 4), (4, 6), (8, 5), (5, 8)
        ]
        unique_patterns = []
        for p in candidate_patterns:
            if p not in unique_patterns:
                unique_patterns.append(p)

        sample_pairs = image_pairs[:min(5, len(image_pairs))]
        for pat in unique_patterns:
            matches = 0
            for img_m, img_u in sample_pairs:
                gray_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY) if len(img_m.shape) == 3 else img_m
                gray_u = cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY) if len(img_u.shape) == 3 else img_u

                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
                ret_m, _ = cv2.findChessboardCorners(gray_m, pat, flags)
                ret_u, _ = cv2.findChessboardCorners(gray_u, pat, flags)

                if ret_m and ret_u:
                    matches += 1

            if matches >= min(2, len(sample_pairs)):
                return pat

        return self.pattern_size

    def detect_corners(self, img: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detects and sub-pixel refines 2D corners in a checkerboard target image using a multi-stage pipeline:
        1. Primary pattern size & transpose test.
        2. Sector-Based Engine (cv2.findChessboardCornersSB) with proper return unpacking.
        3. CLAHE (Contrast Limited Adaptive Histogram Equalization) Preprocessed Detection.
        4. Multi-Scale Image Pyramid Search (1.5x upscaling for distant Ultra-Wide views).
        5. Candidate pattern search (if auto_discover=True).

        :param img: (H, W, 3) BGR or (H, W) grayscale uint8 image array.
        :return: (found_boolean, corners_2d_subpixel, object_points_3d)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        candidate_patterns = [self.pattern_size]
        if self.pattern_size[0] != self.pattern_size[1]:
            candidate_patterns.append((self.pattern_size[1], self.pattern_size[0]))

        if self.auto_discover:
            common_grids = [
                (5, 5), (5, 4), (4, 5), (9, 6), (6, 9), (8, 6), (6, 8),
                (7, 5), (5, 7), (10, 7), (7, 10), (6, 4), (4, 6)
            ]
            for cg in common_grids:
                if cg not in candidate_patterns:
                    candidate_patterns.append(cg)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        image_variants = [(gray, 1.0), (clahe_gray, 1.0)]

        h, w = gray.shape[:2]
        if max(h, w) <= 2048:
            scaled_gray = cv2.resize(gray, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
            image_variants.append((scaled_gray, 1.5))

        for pat_cols, pat_rows in candidate_patterns:
            pattern = (pat_cols, pat_rows)
            for var_img, scale in image_variants:
                # Sector-Based Engine (cv2.findChessboardCornersSB)
                if hasattr(cv2, 'findChessboardCornersSB'):
                    flags_sb = 0
                    if hasattr(cv2, 'CALIB_CB_ACCURACY'):
                        flags_sb |= cv2.CALIB_CB_ACCURACY
                    if hasattr(cv2, 'CALIB_CB_EXHAUSTIVE'):
                        flags_sb |= cv2.CALIB_CB_EXHAUSTIVE

                    try:
                        res = cv2.findChessboardCornersSB(var_img, pattern, flags=flags_sb)
                        if res is not None and isinstance(res, tuple) and len(res) >= 2:
                            found, corners = res[0], res[1]
                            if found and corners is not None and len(corners) == pat_cols * pat_rows:
                                if scale != 1.0:
                                    corners = (corners / scale).astype(np.float32)
                                objp = np.zeros((pat_cols * pat_rows, 3), dtype=np.float32)
                                objp[:, :2] = np.mgrid[0:pat_cols, 0:pat_rows].T.reshape(-1, 2) * self.square_size_m
                                return True, corners, objp
                    except Exception:
                        pass

                # Legacy findChessboardCorners
                flags_legacy = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                found, corners = cv2.findChessboardCorners(var_img, pattern, flags_legacy)

                if not found:
                    flags_filter = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS
                    found, corners = cv2.findChessboardCorners(var_img, pattern, flags_filter)

                if found and corners is not None and len(corners) == pat_cols * pat_rows:
                    if scale != 1.0:
                        corners = (corners / scale).astype(np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    objp = np.zeros((pat_cols * pat_rows, 3), dtype=np.float32)
                    objp[:, :2] = np.mgrid[0:pat_cols, 0:pat_rows].T.reshape(-1, 2) * self.square_size_m
                    return True, corners_subpix, objp

        return False, None, None

    def align_corner_ordering(self, corners1: np.ndarray, corners2: np.ndarray, cols: int, rows: int) -> np.ndarray:
        """
        Ensures corners2 has the exact same corner traversal orientation and origin as corners1.
        If corners2 is inverted (180 degree flip relative to corners1), reverses corners2.
        """
        if corners1 is None or corners2 is None or len(corners1) != cols * rows or len(corners2) != cols * rows:
            return corners2

        p1_start = corners1[0][0]
        p1_end_row = corners1[cols - 1][0]
        p1_end_col = corners1[(rows - 1) * cols][0]

        p2_start = corners2[0][0]
        p2_end_row = corners2[cols - 1][0]
        p2_end_col = corners2[(rows - 1) * cols][0]

        v1_row = p1_end_row - p1_start
        v2_row = p2_end_row - p2_start
        v1_col = p1_end_col - p1_start
        v2_col = p2_end_col - p2_start

        dot_row = float(np.dot(v1_row, v2_row))
        dot_col = float(np.dot(v1_col, v2_col))

        if dot_row < 0.0 or dot_col < 0.0:
            return corners2[::-1].copy()

        return corners2

    def compute_pair_reprojection_errors(
        self,
        obj_points_list: List[np.ndarray],
        img_points1_list: List[np.ndarray],
        img_points2_list: List[np.ndarray],
        K1: np.ndarray, D1: np.ndarray,
        K2: np.ndarray, D2: np.ndarray,
        R: np.ndarray, T: np.ndarray
    ) -> List[float]:
        """
        Computes per-pair RMS reprojection errors across Main & Ultra-Wide views.
        Guards against float overflow and non-finite pose solutions.
        """
        pair_errors = []
        rvec2, _ = cv2.Rodrigues(R)
        tvec2 = T

        for i in range(len(obj_points_list)):
            objp = obj_points_list[i]
            img1 = img_points1_list[i]
            img2 = img_points2_list[i]

            ret1, rvec1_est, tvec1_est = cv2.solvePnP(objp, img1, K1, D1)
            if not ret1 or rvec1_est is None or tvec1_est is None:
                pair_errors.append(999.0)
                continue

            if not (np.all(np.isfinite(rvec1_est)) and np.all(np.isfinite(tvec1_est))):
                pair_errors.append(999.0)
                continue

            proj1, _ = cv2.projectPoints(objp, rvec1_est, tvec1_est, K1, D1)
            pts1 = img1.reshape(-1, 2).astype(np.float64)
            p1 = proj1.reshape(-1, 2).astype(np.float64)

            if not np.all(np.isfinite(p1)):
                err1 = 999.0
            else:
                diff1 = np.asarray(pts1, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
                dist1_sq = np.sum(np.square(diff1, dtype=np.float64), axis=1)
                err1 = float(np.sqrt(np.mean(dist1_sq)))
                if not np.isfinite(err1) or err1 > 999.0:
                    err1 = 999.0

            R_target1, _ = cv2.Rodrigues(rvec1_est)
            R_target2 = R @ R_target1
            T_target2 = R @ tvec1_est + tvec2

            proj2, _ = cv2.projectPoints(objp, R_target2, T_target2, K2, D2)
            pts2 = img2.reshape(-1, 2).astype(np.float64)
            p2 = proj2.reshape(-1, 2).astype(np.float64)

            if not np.all(np.isfinite(p2)):
                err2 = 999.0
            else:
                diff2 = np.asarray(pts2, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
                dist2_sq = np.sum(np.square(diff2, dtype=np.float64), axis=1)
                err2 = float(np.sqrt(np.mean(dist2_sq)))
                if not np.isfinite(err2) or err2 > 999.0:
                    err2 = 999.0

            total_err = float((err1 + err2) / 2.0)
            if not np.isfinite(total_err) or total_err > 999.0:
                total_err = 999.0

            pair_errors.append(total_err)

        return pair_errors

    def calibrate_stereo(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        initial_K1: np.ndarray,
        initial_K2: np.ndarray,
        initial_D1: Optional[np.ndarray] = None,
        initial_D2: Optional[np.ndarray] = None,
        verbose: bool = True,
        max_reproj_error_threshold: float = 1.2,
        max_outlier_prune_iters: int = 3
    ) -> Dict[str, Any]:
        """
        Runs robust 3-stage stereo calibration across snapshot calibration pairs:
        1. Multi-stage pattern detection & alignment with automatic pattern discovery.
        2. Initial camera pre-calibration & stereo bundle adjustment solver.
        3. Iterative outlier pruning loop discarding high reprojection error pairs.
        """
        if self.auto_discover:
            discovered_pattern = self.auto_discover_pattern_size(image_pairs)
            if discovered_pattern != self.pattern_size:
                if verbose:
                    print(f"[CalibRefiner] Auto-discovered pattern size: {discovered_pattern} (was {self.pattern_size})")
                self.pattern_size = discovered_pattern
                cols, rows = discovered_pattern
                self.objp = np.zeros((cols * rows, 3), dtype=np.float32)
                self.objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size_m
            self.auto_discover = False

        obj_points_list = []
        img_points1_list = []
        img_points2_list = []
        pair_indices = []

        h, w = image_pairs[0][0].shape[:2]

        for idx, (img_main, img_uw) in enumerate(image_pairs):
            found1, corners1, obj1 = self.detect_corners(img_main)
            found2, corners2, obj2 = self.detect_corners(img_uw)

            if verbose:
                status1 = "FOUND" if found1 else "NOT FOUND"
                status2 = "FOUND" if found2 else "NOT FOUND"
                print(f"[CalibRefiner] Pair #{idx+1:02d} -> Main: {status1} | Ultra-Wide: {status2}")

            if found1 and found2 and obj1 is not None and obj2 is not None and obj1.shape == obj2.shape:
                cols = self.pattern_size[0]
                rows = self.pattern_size[1]
                corners2_aligned = self.align_corner_ordering(corners1, corners2, cols, rows)

                obj_points_list.append(obj1)
                img_points1_list.append(corners1)
                img_points2_list.append(corners2_aligned)
                pair_indices.append(idx)

        # Fallback if no pairs detected with current pattern_size, try auto-discovery
        if len(obj_points_list) < 1 and not self.auto_discover:
            discovered_pattern = self.auto_discover_pattern_size(image_pairs)
            if discovered_pattern != self.pattern_size:
                if verbose:
                    print(f"[CalibRefiner] Retrying with auto-discovered pattern size: {discovered_pattern}")
                self.pattern_size = discovered_pattern
                cols, rows = discovered_pattern
                self.objp = np.zeros((cols * rows, 3), dtype=np.float32)
                self.objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size_m

                for idx, (img_main, img_uw) in enumerate(image_pairs):
                    found1, corners1, obj1 = self.detect_corners(img_main)
                    found2, corners2, obj2 = self.detect_corners(img_uw)
                    if found1 and found2 and obj1 is not None and obj2 is not None and obj1.shape == obj2.shape:
                        corners2_aligned = self.align_corner_ordering(corners1, corners2, discovered_pattern[0], discovered_pattern[1])
                        obj_points_list.append(obj1)
                        img_points1_list.append(corners1)
                        img_points2_list.append(corners2_aligned)
                        pair_indices.append(idx)

        D1_init = initial_D1.copy() if initial_D1 is not None else np.zeros(5, dtype=np.float64)
        D2_init = initial_D2.copy() if initial_D2 is not None else np.zeros(5, dtype=np.float64)

        if len(obj_points_list) < 1:
            if verbose:
                print("\n[CalibRefiner] ⚠️ WARNING: No snapshot pairs had valid checkerboards detected in BOTH Main & Ultra-Wide views!")
            return {
                "K1": initial_K1,
                "D1": D1_init,
                "K2": initial_K2,
                "D2": D2_init,
                "R": np.eye(3, dtype=np.float64),
                "T": np.array([[-0.0195], [0.0], [0.0]], dtype=np.float64),
                "reprojection_error": 0.0,
                "num_pairs_used": 0,
                "per_pair_errors": [],
                "inlier_indices": []
            }

        # Stage 1: Individual camera pre-calibration for well-conditioned intrinsic initial estimates
        K1_pre = initial_K1.copy()
        D1_pre = D1_init.copy()
        K2_pre = initial_K2.copy()
        D2_pre = D2_init.copy()

        try:
            _, K1_pre, D1_pre, _, _ = cv2.calibrateCamera(
                obj_points_list, img_points1_list, (w, h),
                initial_K1.copy(), D1_init,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
        except cv2.error as e:
            if verbose:
                print(f"[CalibRefiner] Stage 1 pre-calibration warning (cam1): {e}")

        try:
            _, K2_pre, D2_pre, _, _ = cv2.calibrateCamera(
                obj_points_list, img_points2_list, (w, h),
                initial_K2.copy(), D2_init,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
        except cv2.error as e:
            if verbose:
                print(f"[CalibRefiner] Stage 1 pre-calibration warning (cam2): {e}")

        # Stage 2: Stereo Optimization Solver
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret, K1_ref, D1_ref, K2_ref, D2_ref, R_ref, T_ref, E, F = cv2.stereoCalibrate(
            objectPoints=obj_points_list,
            imagePoints1=img_points1_list,
            imagePoints2=img_points2_list,
            cameraMatrix1=K1_pre, distCoeffs1=D1_pre,
            cameraMatrix2=K2_pre, distCoeffs2=D2_pre,
            imageSize=(w, h),
            flags=flags,
            criteria=criteria
        )

        # Stage 3: Iterative Outlier Pruning Loop
        curr_obj = list(obj_points_list)
        curr_img1 = list(img_points1_list)
        curr_img2 = list(img_points2_list)
        curr_pairs = list(pair_indices)
        pair_errors = self.compute_pair_reprojection_errors(
            curr_obj, curr_img1, curr_img2, K1_ref, D1_ref, K2_ref, D2_ref, R_ref, T_ref
        )

        for iter_idx in range(max_outlier_prune_iters):
            if len(curr_obj) <= 4:
                break
            max_err_idx = int(np.argmax(pair_errors))
            max_err = pair_errors[max_err_idx]

            if max_err > max_reproj_error_threshold:
                if verbose:
                    print(f"[CalibRefiner] Outlier Pruning Iteration #{iter_idx+1}: Removing pair #{curr_pairs[max_err_idx]+1} (reprojection error {max_err:.4f} px > {max_reproj_error_threshold} px)")
                curr_obj.pop(max_err_idx)
                curr_img1.pop(max_err_idx)
                curr_img2.pop(max_err_idx)
                curr_pairs.pop(max_err_idx)

                ret, K1_ref, D1_ref, K2_ref, D2_ref, R_ref, T_ref, E, F = cv2.stereoCalibrate(
                    objectPoints=curr_obj,
                    imagePoints1=curr_img1,
                    imagePoints2=curr_img2,
                    cameraMatrix1=K1_ref, distCoeffs1=D1_ref,
                    cameraMatrix2=K2_ref, distCoeffs2=D2_ref,
                    imageSize=(w, h),
                    flags=flags,
                    criteria=criteria
                )
                pair_errors = self.compute_pair_reprojection_errors(
                    curr_obj, curr_img1, curr_img2, K1_ref, D1_ref, K2_ref, D2_ref, R_ref, T_ref
                )
            else:
                break

        return {
            "K1": K1_ref,
            "D1": D1_ref,
            "K2": K2_ref,
            "D2": D2_ref,
            "R": R_ref,
            "T": T_ref,
            "reprojection_error": float(ret),
            "num_pairs_used": len(curr_obj),
            "per_pair_errors": pair_errors,
            "inlier_indices": curr_pairs,
            "pattern_size": self.pattern_size
        }
