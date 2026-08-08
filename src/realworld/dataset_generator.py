"""
Synthetic Heterogeneous Dual-Camera & RealSense Benchmark Dataset Generator.
Generates synthetic Main RGB, Ultra-Wide RGB, per-frame dynamic calibration JSON, and RealSense GT depth/pointcloud files.
"""

import os
import json
import numpy as np
import cv2


def generate_synthetic_realworld_dataset(output_dir: str = "data/realworld_sample"):
    """
    Generates a synthetic heterogeneous dataset conforming to dataset_manifest.json schema.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_dir = os.path.join(output_dir, "frame_000000")
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Dimensions and focal length setup
    W_main, H_main = 1920, 1080
    W_uw, H_uw = 1920, 1080

    # Focal lengths: Main ~ 1400 px, UltraWide ~ 600 px
    f_main = 1400.0
    f_uw = 600.0

    K1 = np.array([
        [f_main, 0.0, W_main / 2.0],
        [0.0, f_main, H_main / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    K2 = np.array([
        [f_uw, 0.0, W_uw / 2.0],
        [0.0, f_uw, H_uw / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    R = np.eye(3, dtype=np.float64)
    T = np.array([19.5, 0.0, 0.0], dtype=np.float64)  # 19.5 mm baseline

    # 2. Synthesize 3D Scene Geometry (Background wall at Z=2.0m, Foreground Box at Z=1.0m)
    u_grid, v_grid = np.meshgrid(np.arange(W_main), np.arange(H_main))
    gt_depth = np.ones((H_main, W_main), dtype=np.float32) * 2.0  # 2 meters background

    # Place a central foreground box at Z = 1.0m
    box_mask = (u_grid > W_main * 0.35) & (u_grid < W_main * 0.65) & (v_grid > H_main * 0.3) & (v_grid < H_main * 0.7)
    gt_depth[box_mask] = 1.0

    # 3. Synthesize Main RGB Image (Textured checker pattern)
    checker = ((u_grid // 32) % 2) ^ ((v_grid // 32) % 2)
    img_main = np.zeros((H_main, W_main, 3), dtype=np.uint8)
    img_main[:, :, 0] = np.where(checker, 200, 50)
    img_main[:, :, 1] = np.where(box_mask, 180, 80)
    img_main[:, :, 2] = np.where(checker, 80, 200)

    # 4. Synthesize Ultra-Wide RGB Image via backward projection
    # Shift Ultra-Wide according to baseline and focal length ratio
    img_uw = np.zeros((H_uw, W_uw, 3), dtype=np.uint8)
    baseline_m = 0.0195
    disparity_main = (f_main * baseline_m) / gt_depth

    # Unproject Main to 3D and project to Ultra-Wide
    X_3d = (u_grid - W_main / 2.0) * gt_depth / f_main
    Y_3d = (v_grid - H_main / 2.0) * gt_depth / f_main
    Z_3d = gt_depth

    # Transform to UW frame (X_uw = X - T_x)
    X_uw = X_3d - baseline_m
    u_uw_proj = np.round(f_uw * (X_uw / Z_3d) + W_uw / 2.0).astype(np.int32)
    v_uw_proj = np.round(f_uw * (Y_3d / Z_3d) + H_uw / 2.0).astype(np.int32)

    valid_uw = (u_uw_proj >= 0) & (u_uw_proj < W_uw) & (v_uw_proj >= 0) & (v_uw_proj < H_uw)
    img_uw[v_uw_proj[valid_uw], u_uw_proj[valid_uw]] = img_main[v_grid[valid_uw], u_grid[valid_uw]]
    # Fill unmapped regions with subtle noise
    blank_uw = (img_uw[:, :, 0] == 0) & (img_uw[:, :, 1] == 0) & (img_uw[:, :, 2] == 0)
    img_uw[blank_uw] = 100

    # 5. Save Image & Ground Truth Files
    main_rgb_rel = "frame_000000/im0_main.png"
    uw_rgb_rel = "frame_000000/im1_ultrawide.png"
    rs_depth_png_rel = "frame_000000/realsense_depth.png"
    rs_depth_npy_rel = "frame_000000/realsense_depth.npy"
    rs_ply_rel = "frame_000000/realsense_gt.ply"

    cv2.imwrite(os.path.join(output_dir, main_rgb_rel), img_main)
    cv2.imwrite(os.path.join(output_dir, uw_rgb_rel), img_uw)

    # Save RealSense depth map (uint16 mm PNG and float32 meters NPY)
    np.save(os.path.join(output_dir, rs_depth_npy_rel), gt_depth)
    cv2.imwrite(os.path.join(output_dir, rs_depth_png_rel), (gt_depth * 1000.0).astype(np.uint16))

    # Save PLY point cloud
    pts_valid = gt_depth > 0.1
    x_pts = X_3d[pts_valid]
    y_pts = Y_3d[pts_valid]
    z_pts = Z_3d[pts_valid]
    pts = np.vstack((x_pts, y_pts, z_pts)).T

    ply_path = os.path.join(output_dir, rs_ply_rel)
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for pt in pts[::10]:  # Downsample for lightweight test artifact
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")

    # 6. Construct JSON Manifest
    manifest = {
        "sequence_id": "seq_synthetic_sample_001",
        "version": "1.0.0",
        "created_at": "2026-08-08T16:51:00Z",
        "devices": {
            "iphone": {
                "model": "iPhone 15 Pro",
                "os_version": "iOS 17.4",
                "nominal_baseline_mm": 19.5,
                "main_camera": {"sensor_name": "Main Wide 24mm", "nominal_focal_length_mm": 6.86, "nominal_fov_deg": 80.0},
                "ultrawide_camera": {"sensor_name": "Ultra-Wide 13mm", "nominal_focal_length_mm": 2.22, "nominal_fov_deg": 120.0}
            },
            "realsense": {
                "model": "D435",
                "serial_number": "12345678",
                "firmware_version": "5.13.0",
                "baseline_mm": 50.0,
                "depth_units_meters": 0.001
            }
        },
        "frame_count": 1,
        "frames": [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "files": {
                    "main_rgb": main_rgb_rel,
                    "ultrawide_rgb": uw_rgb_rel,
                    "realsense_depth_png": rs_depth_png_rel,
                    "realsense_depth_npy": rs_depth_npy_rel,
                    "realsense_pointcloud_ply": rs_ply_rel
                },
                "iphone_calibration": {
                    "reference_dimensions": {"width": W_main, "height": H_main},
                    "main_intrinsics": {
                        "matrix_3x3": K1.tolist(),
                        "pixel_size_micrometers": 1.22,
                        "lens_position": 0.5
                    },
                    "ultrawide_intrinsics": {
                        "matrix_3x3": K2.tolist(),
                        "pixel_size_micrometers": 1.0,
                        "lens_position": 0.0
                    },
                    "extrinsic_transform_ultrawide_to_main": {
                        "rotation_matrix_3x3": R.tolist(),
                        "translation_vector_mm": T.tolist()
                    }
                },
                "realsense_calibration": {
                    "dimensions": {"width": W_main, "height": H_main},
                    "intrinsics_matrix_3x3": K1.tolist(),
                    "depth_scale_meters": 0.001
                }
            }
        ]
    }

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated synthetic realworld dataset at {output_dir}")
    return output_dir


if __name__ == "__main__":
    generate_synthetic_realworld_dataset()
