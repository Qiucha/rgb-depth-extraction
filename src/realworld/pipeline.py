"""
Master Execution Pipeline for Real-World iPhone Dual-Camera & Intel RealSense Stereo Depth Evaluation.
Integrates heterogeneous rectification, block matching, WLS filtering, ICP registration, and digest report generation.
"""

import os
import json
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .hetero_rectifier import HeterogeneousStereoRectifier
from .dataset_loader import RealWorldDatasetLoader
from .realsense_icp import RealSensePointcloudAligner
from .deep_matcher import get_stereo_matcher, BaseStereoMatcher
from .evaluator import RealWorldEvaluator
from .digest_builder import generate_realworld_digest

from src.stereo_depth.wls_filter import WLSDisparityFilter
from src.stereo_depth.depth_calculator import DepthCalculator
from src.stereo_depth.digest_generator import generate_digest_data


def run_realworld_pipeline(
    sequence_dir: str,
    output_dir: str = "digest_realworld",
    target_size: tuple = (1280, 960),
    matcher_type: str = "sliding_window",
    model_path: Optional[str] = None,
    matcher_kwargs: Optional[dict] = None
):
    """
    Executes end-to-end realworld evaluation pipeline on a captured stereo pair sequence directory.

    :param sequence_dir: Path to folder containing dataset_manifest.json and frames.
    :param output_dir: Path to directory where HTML digest telemetry will be generated.
    :param target_size: (width, height) resolution for rectification & block matching.
    :param matcher_type: Stereo matching algorithm backend ('sliding_window', 'cre_stereo', 'raft_stereo', 'anystereo')
    :param model_path: Optional path to deep learning model weights (.onnx, .pt, .pth).
    :param matcher_kwargs: Optional additional dictionary arguments for stereo matcher setup.
    :return: Summary dictionary containing evaluation metrics and telemetry paths.
    """
    print(f"=== Running Real-World iPhone Stereo Pipeline on {sequence_dir} [Matcher: {matcher_type}] ===")

    # 1. Load sequence dataset manifest
    loader = RealWorldDatasetLoader(sequence_dir)
    frame_data = loader.load_frame(0)

    # 2. Perform FOV alignment and dynamic epipolar rectification
    rectifier = HeterogeneousStereoRectifier(target_size=target_size)
    rect_main, rect_uw, P1, P2, Q = rectifier.rectify_pair(
        img_main=frame_data["img_main"],
        img_uw=frame_data["img_uw"],
        K1=frame_data["K1"], D1=frame_data["D1"],
        K2=frame_data["K2"], D2=frame_data["D2"],
        R=frame_data["R"], T=frame_data["T"]
    )

    # 3. Execute Stereo Matching (SlidingWindowMatcher or Deep Disparity Adapter)
    extra_kwargs = matcher_kwargs or {}
    if model_path is not None and "model_path" not in extra_kwargs:
        extra_kwargs["model_path"] = model_path

    matcher = get_stereo_matcher(matcher_name=matcher_type, **extra_kwargs)
    raw_disparity, valid_mask = matcher.compute_disparity(rect_main, rect_uw)

    # Convert main to grayscale for WLS filtering
    gray_main = cv2.cvtColor(rect_main, cv2.COLOR_BGR2GRAY) if rect_main.ndim == 3 else rect_main

    # 4. Apply Weighted Least Squares (WLS) Edge-Preserving Filter
    wls = WLSDisparityFilter(lambda_val=8000.0, sigma_val=1.5)
    filtered_disparity = wls.filter(raw_disparity, gray_main)

    # 5. Convert disparity to physical metric depth (Z = f * B / d)
    f_rect = P1[0, 0]
    baseline_m = float(np.linalg.norm(frame_data["T"])) / 1000.0 if np.linalg.norm(frame_data["T"]) > 1.0 else float(np.linalg.norm(frame_data["T"]))
    
    depth_calc = DepthCalculator(focal_length=f_rect, baseline=baseline_m, doffs=0.0)
    depth_map_est = depth_calc.disparity_to_depth(filtered_disparity)

    # 6. Point Cloud ICP Registration against RealSense Ground Truth (if available)
    icp_results = None
    if frame_data["realsense_depth"] is not None:
        aligner = RealSensePointcloudAligner()
        src_pts, _ = aligner.depth_to_pointcloud(depth_map_est, P1, rect_main)
        
        # Resize GT depth map to match target working resolution if needed
        rs_gt_depth = frame_data["realsense_depth"]
        if rs_gt_depth.shape != target_size[::-1]:
            rs_gt_depth = cv2.resize(rs_gt_depth, target_size, interpolation=cv2.INTER_NEAREST)

        tgt_pts, _ = aligner.depth_to_pointcloud(rs_gt_depth, P1, rect_main)

        if len(src_pts) > 10 and len(tgt_pts) > 10:
            aligned_pts, transform_mat, fitness, inlier_rmse = aligner.align_icp(src_pts, tgt_pts)
            icp_results = {
                "fitness": float(fitness),
                "inlier_rmse_m": float(inlier_rmse),
                "transform": transform_mat.tolist()
            }

        # 7. Evaluate iPhone depth error against RealSense ground truth
        evaluator = RealWorldEvaluator()
        eval_metrics = evaluator.evaluate_all(depth_map_est, rs_gt_depth, rect_main)
    else:
        eval_metrics = {
            "mae_m": 0.0,
            "rmse_m": 0.0,
            "bad_pixel_ratio": 0.0,
            "texture_dependency_ratio": 1.0,
            "flying_pixel_ratio": 0.0
        }

    # 8. Generate Telemetry & Visual Digest Data (HTML Dashboard, 3D Point Cloud, Overlays)
    os.makedirs(output_dir, exist_ok=True)
    generate_realworld_digest(
        rect_main=rect_main,
        rect_uw=rect_uw,
        disparity_map=filtered_disparity,
        depth_map_m=depth_map_est,
        focal_length_px=float(f_rect),
        baseline_m=float(baseline_m),
        output_dir=output_dir,
        scene_name=os.path.basename(sequence_dir) or "iPhone Stereo Capture",
        eval_metrics=eval_metrics
    )

    summary_path = os.path.join(output_dir, "realworld_summary.json")
    summary_data = {
        "sequence_dir": sequence_dir,
        "target_resolution": target_size,
        "rectified_focal_length_px": float(f_rect),
        "baseline_meters": float(baseline_m),
        "evaluation_metrics": eval_metrics,
        "icp_registration": icp_results
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Successfully finished realworld pipeline! Digest generated at {output_dir}/index.html")
    return summary_data
