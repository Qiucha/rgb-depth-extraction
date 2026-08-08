"""
Dataset Loader for iPhone Dual-Camera Realworld Captures and Intel RealSense Ground Truth.
Provides JSON manifest parsing and synthesized calib.txt adapter for Middlebury compatibility.
"""

import os
import json
import math
import cv2
import numpy as np


class RealWorldDatasetLoader:
    """
    Ingests sequence directory conforming to the JSON schema in Issue #2,
    loading RGB images, per-frame dynamic calibration matrices, and RealSense ground truth.
    """

    def __init__(self, sequence_dir: str):
        self.sequence_dir = sequence_dir
        self.manifest_path = os.path.join(sequence_dir, "dataset_manifest.json")
        self.manifest = None
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)

    def load_frame(self, frame_index: int = 0):
        """
        Loads frame data by index. Returns a dictionary containing RGB images, calibration matrices,
        and RealSense ground truth depth map / point cloud path.
        """
        if self.manifest and "frames" in self.manifest:
            frame_info = self.manifest["frames"][frame_index]
            files = frame_info["files"]

            img_main_path = os.path.join(self.sequence_dir, files["main_rgb"])
            img_uw_path = os.path.join(self.sequence_dir, files["ultrawide_rgb"])

            img_main = cv2.imread(img_main_path)
            img_uw = cv2.imread(img_uw_path)

            iphone_calib = frame_info["iphone_calibration"]
            K1 = np.array(iphone_calib["main_intrinsics"]["matrix_3x3"], dtype=np.float64)
            K2 = np.array(iphone_calib["ultrawide_intrinsics"]["matrix_3x3"], dtype=np.float64)

            ext = iphone_calib["extrinsic_transform_ultrawide_to_main"]
            R = np.array(ext["rotation_matrix_3x3"], dtype=np.float64)
            T = np.array(ext["translation_vector_mm"], dtype=np.float64)

            rs_depth = None
            rs_depth_path = os.path.join(self.sequence_dir, files.get("realsense_depth_npy", ""))
            if os.path.exists(rs_depth_path):
                rs_depth = np.load(rs_depth_path)

            return {
                "img_main": img_main,
                "img_uw": img_uw,
                "K1": K1,
                "K2": K2,
                "D1": np.zeros(5, dtype=np.float64),
                "D2": np.zeros(5, dtype=np.float64),
                "R": R,
                "T": T,
                "realsense_depth": rs_depth,
                "realsense_ply_path": os.path.join(self.sequence_dir, files.get("realsense_pointcloud_ply", ""))
            }
        else:
            raise FileNotFoundError(f"dataset_manifest.json not found in {self.sequence_dir}")

    def export_middlebury_adapter(self, frame_data: dict, output_dir: str):
        """
        Converts frame calibration data into Middlebury-style calib.txt for src/stereo_depth/ compatibility.
        """
        K1 = frame_data["K1"]
        T = frame_data["T"]
        baseline_m = float(np.linalg.norm(T)) / 1000.0 if np.linalg.norm(T) > 1.0 else float(np.linalg.norm(T))

        cam0_str = f"[{K1[0,0]:.4f} 0 {K1[0,2]:.4f}; 0 {K1[1,1]:.4f} {K1[1,2]:.4f}; 0 0 1]"
        calib_content = (
            f"cam0={cam0_str}\n"
            f"cam1={cam0_str}\n"
            f"doffs=0.0\n"
            f"baseline={baseline_m:.6f}\n"
            f"width={frame_data['img_main'].shape[1]}\n"
            f"height={frame_data['img_main'].shape[0]}\n"
            f"ndisp=128\n"
            f"vmin=0\n"
            f"vmax=255\n"
        )
        os.makedirs(output_dir, exist_ok=True)
        calib_path = os.path.join(output_dir, "calib.txt")
        with open(calib_path, "w") as f:
            f.write(calib_content)
        return calib_path
