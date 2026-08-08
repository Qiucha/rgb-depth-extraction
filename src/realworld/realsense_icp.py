"""
Open3D Iterative Closest Point (ICP) Alignment Engine.
Aligns 3D point cloud reconstructed from iPhone passive stereo depth maps with Intel RealSense ground-truth point clouds.
"""

import numpy as np


class RealSensePointcloudAligner:
    """
    Registers iPhone passive stereo point cloud to Intel RealSense active IR ground-truth point cloud.
    """

    def __init__(self, max_correspondence_distance: float = 0.05):
        """
        :param max_correspondence_distance: Maximum distance threshold for ICP point matching in meters.
        """
        self.max_correspondence_distance = max_correspondence_distance

    def depth_to_pointcloud(self, depth_map: np.ndarray, P1: np.ndarray, rgb_img: np.ndarray = None):
        """
        Unprojects depth map Z (in meters) into 3D point cloud array (N, 3) using rectified projection matrix P1.
        """
        fx = P1[0, 0]
        fy = P1[1, 1]
        cx = P1[0, 2]
        cy = P1[1, 2]

        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        valid_mask = (depth_map > 0.1) & (depth_map < 10.0) & np.isfinite(depth_map)
        z_valid = depth_map[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        x_valid = (u_valid - cx) * z_valid / fx
        y_valid = (v_valid - cy) * z_valid / fy

        pts = np.vstack((x_valid, y_valid, z_valid)).T
        colors = None
        if rgb_img is not None:
            colors = rgb_img[valid_mask] / 255.0

        return pts, colors

    def align_icp(self, source_pts: np.ndarray, target_pts: np.ndarray, init_transform: np.ndarray = None):
        """
        Performs Point-to-Point ICP registration using Open3D (if available) or rigid SVD fallback.

        :param source_pts: (N, 3) iPhone point cloud array in meters.
        :param target_pts: (M, 3) RealSense point cloud array in meters.
        :param init_transform: Initial 4x4 rigid matrix transform guess.
        :return: (aligned_pts, transformation_matrix, fitness, inlier_rmse)
        """
        if init_transform is None:
            init_transform = np.eye(4)

        try:
            import open3d as o3d

            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_pts)

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_pts)

            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, self.max_correspondence_distance, init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            aligned_pcd = source_pcd.transform(reg_p2p.transformation)
            aligned_pts = np.asarray(aligned_pcd.points)

            return aligned_pts, reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

        except ImportError:
            # SVD fallback alignment when Open3D is not installed in lightweight python environments
            centroid_src = np.mean(source_pts, axis=0)
            centroid_tgt = np.mean(target_pts, axis=0)

            src_centered = source_pts - centroid_src
            tgt_centered = target_pts[:len(source_pts)] - centroid_tgt

            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            t = centroid_tgt - R @ centroid_src

            T_mat = np.eye(4)
            T_mat[:3, :3] = R
            T_mat[:3, 3] = t

            aligned_pts = (R @ source_pts.T).T + t
            diff = aligned_pts - target_pts[:len(aligned_pts)]
            rmse = float(np.sqrt(np.mean(diff ** 2)))

            return aligned_pts, T_mat, 1.0, rmse
