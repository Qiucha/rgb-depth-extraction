"""
Depth calculation and 3D spatial point cloud projection module.
"""

import numpy as np


class DepthCalculator:
    def __init__(self, focal_length, baseline, doffs=0.0, cx=None, cy=None):
        """
        :param focal_length: Focal length f in pixels
        :param baseline: Baseline distance B (mm or meters)
        :param doffs: Disparity offset (pixels)
        :param cx: Principal point X coordinate
        :param cy: Principal point Y coordinate
        """
        self.focal_length = focal_length
        self.baseline = baseline
        self.doffs = doffs
        self.cx = cx
        self.cy = cy

    def disparity_to_depth(self, disparity_map, min_depth=0.1, max_depth=100.0):
        """
        Converts disparity map (pixels) to metric depth map Z = (f * B) / (d + doffs).
        Returns depth_map in same metric units as baseline.
        """
        denom = disparity_map + self.doffs
        # Mask out zero or negative disparity values
        valid_mask = (denom > 1e-3)

        depth_map = np.zeros_like(disparity_map, dtype=np.float32)
        depth_map[valid_mask] = (self.focal_length * self.baseline) / denom[valid_mask]

        # Clip unfeasible extreme depth values
        depth_map[depth_map > max_depth] = max_depth
        depth_map[depth_map < min_depth] = 0.0

        return depth_map

    def generate_point_cloud(self, depth_map, rgb_image=None, subsample_step=4, max_points=20000):
        """
        Generates 3D spatial points (X, Y, Z, R, G, B) from depth map.
        """
        h, w = depth_map.shape
        cx = self.cx if self.cx is not None else w / 2.0
        cy = self.cy if self.cy is not None else h / 2.0

        # Subsample grid for interactive 3D rendering
        y_indices, x_indices = np.mgrid[0:h:subsample_step, 0:w:subsample_step]

        sub_depth = depth_map[y_indices, x_indices]
        valid = (sub_depth > 0) & np.isfinite(sub_depth)

        x_vals = x_indices[valid]
        y_vals = y_indices[valid]
        z_vals = sub_depth[valid]

        # Project 2D pixel coordinates to 3D world coordinates
        X = (x_vals - cx) * z_vals / self.focal_length
        Y = (y_vals - cy) * z_vals / self.focal_length
        Z = z_vals

        points = np.column_stack((X, Y, Z))

        if rgb_image is not None:
            sub_rgb = rgb_image[y_indices, x_indices]
            colors = sub_rgb[valid]
            # Convert BGR to RGB if needed
            if colors.shape[1] == 3:
                colors = colors[:, [2, 1, 0]]
        else:
            colors = np.ones((len(points), 3), dtype=np.uint8) * 255

        # Limit maximum points for web rendering performance
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        return points, colors
