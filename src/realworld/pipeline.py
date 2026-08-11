"""
Master Execution Pipeline for Real-World iPhone Dual-Camera & Intel RealSense Stereo Depth Evaluation.
Integrates color/exposure normalization, heterogeneous fisheye rectification, Census-SGBM matching,
multi-stage disparity post-processing, ICP registration, and visual digest report generation.
"""

import os
import json
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .hetero_rectifier import HeterogeneousStereoRectifier
from .color_normalizer import ColorExposureNormalizer
from .acutance_enhancer import LanczosAcutanceEnhancer
from .disparity_postprocessor import DisparityPostProcessor
from .dataset_loader import RealWorldDatasetLoader
from .realsense_icp import RealSensePointcloudAligner
from .deep_matcher import get_stereo_matcher, BaseStereoMatcher
from .evaluator import RealWorldEvaluator
from .digest_builder import generate_realworld_digest
from .quality_gates import (
    validate_calibration, validate_rectification,
    validate_disparity, validate_depth
)

from src.stereo_depth.depth_calculator import DepthCalculator


def run_realworld_pipeline(
    sequence_dir: str,
    output_dir: str = "digest_realworld",
    target_size: tuple = (1280, 960),
    matcher_type: str = "census_sgbm",
    model_path: Optional[str] = None,
    matcher_kwargs: Optional[dict] = None,
    is_fisheye: bool = False,
    enhance_acutance: bool = True,
    refined_calib: Optional[dict] = None,
    strict_quality_gates: bool = True
):
    """
    Executes end-to-end realworld evaluation pipeline on a captured stereo pair sequence directory.

    :param sequence_dir: Path to folder containing dataset_manifest.json and frames.
    :param output_dir: Path to directory where HTML digest telemetry will be generated.
    :param target_size: (width, height) resolution for rectification & block matching.
    :param matcher_type: Stereo matching algorithm backend ('census_sgbm', 'sliding_window', 'cre_stereo', 'raft_stereo', 'anystereo')
    :param model_path: Optional path to deep learning model weights (.onnx, .pt, .pth).
    :param matcher_kwargs: Optional additional dictionary arguments for stereo matcher setup.
    :param is_fisheye: If True, enables cv2.fisheye stereo rectification model.
    :param enhance_acutance: If True, applies LanczosAcutanceEnhancer to boost edge acutance.
    :param refined_calib: Optional dictionary containing refined K1, D1, K2, D2, R, T from CalibrationRefiner.
    :param strict_quality_gates: If True, raises exceptions when quality metrics fall below thresholds.
    :return: Summary dictionary containing evaluation metrics and telemetry paths.
    """
    print(f"=== Running Real-World iPhone Stereo Pipeline on {sequence_dir} [Matcher: {matcher_type}] ===")

    # 1. Load sequence dataset manifest
    loader = RealWorldDatasetLoader(sequence_dir)
    frame_data = loader.load_frame(0)

    if refined_calib is None:
        candidates = [
            os.path.join(sequence_dir, "calibration_refined.json"),
            os.path.join(sequence_dir, "calib_refined.json")
        ]
        for cand in candidates:
            if os.path.exists(cand):
                try:
                    with open(cand, "r") as f:
                        data = json.load(f)
                    calib_dict = data.get("iphone_calibration", data)
                    refined_calib = {
                        "K1": np.array(calib_dict.get("K1", calib_dict.get("main_intrinsics", {}).get("matrix_3x3", frame_data["K1"])), dtype=np.float64),
                        "D1": np.array(calib_dict.get("D1", frame_data["D1"]), dtype=np.float64),
                        "K2": np.array(calib_dict.get("K2", calib_dict.get("ultrawide_intrinsics", {}).get("matrix_3x3", frame_data["K2"])), dtype=np.float64),
                        "D2": np.array(calib_dict.get("D2", frame_data["D2"]), dtype=np.float64),
                        "R": np.array(calib_dict.get("R", calib_dict.get("extrinsic_transform_ultrawide_to_main", {}).get("rotation_matrix_3x3", frame_data["R"])), dtype=np.float64),
                        "T": np.array(calib_dict.get("T", calib_dict.get("extrinsic_transform_ultrawide_to_main", {}).get("translation_vector_mm", frame_data["T"])), dtype=np.float64)
                    }
                    if refined_calib["R"].shape == (3, 1) or refined_calib["R"].shape == (1, 3):
                        refined_calib["R"], _ = cv2.Rodrigues(refined_calib["R"])
                    refined_calib["T"] = refined_calib["T"].flatten()
                    print(f"[Pipeline] Auto-applied refined calibration from '{cand}'")
                    break
                except Exception as e:
                    print(f"[Pipeline] Warning: Failed to parse calibration file {cand}: {e}")

    K1 = refined_calib["K1"] if refined_calib and "K1" in refined_calib else frame_data["K1"]
    D1 = refined_calib["D1"] if refined_calib and "D1" in refined_calib else frame_data["D1"]
    K2 = refined_calib["K2"] if refined_calib and "K2" in refined_calib else frame_data["K2"]
    D2 = refined_calib["D2"] if refined_calib and "D2" in refined_calib else frame_data["D2"]
    R = refined_calib["R"] if refined_calib and "R" in refined_calib else frame_data["R"]
    T = refined_calib["T"] if refined_calib and "T" in refined_calib else frame_data["T"]

    # Quality Gate 1: Validate calibration parameters
    if strict_quality_gates:
        reprojection_error = refined_calib.get("reprojection_error") if refined_calib else None
        validate_calibration(R, T, reprojection_error=reprojection_error)

    # 2. Perform FOV alignment and dynamic epipolar rectification
    rectifier = HeterogeneousStereoRectifier(target_size=target_size, is_fisheye=is_fisheye)
    rect_main, rect_uw, P1, P2, Q = rectifier.rectify_pair(
        img_main=frame_data["img_main"],
        img_uw=frame_data["img_uw"],
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T
    )

    # Quality Gate 2: Validate rectification output
    if strict_quality_gates:
        validate_rectification(rect_main, rect_uw)


    # 3. Perform Color & Exposure Luminance Normalization
    normalizer = ColorExposureNormalizer(method="cdf")
    rect_uw_norm = normalizer.match_luminance(rect_uw, rect_main)

    # 3b. Apply Adaptive Unsharp Masking Acutance Enhancement to align high-frequency edge gradients
    if enhance_acutance:
        enhancer = LanczosAcutanceEnhancer(amount=1.5, radius=1.0, threshold=0)
        rect_main = enhancer.enhance(rect_main)
        rect_uw_norm = enhancer.enhance(rect_uw_norm)


    # 4. Execute Stereo Matching (Census-SGBM or Deep Disparity Adapter)
    extra_kwargs = matcher_kwargs or {}
    if model_path is not None and "model_path" not in extra_kwargs:
        extra_kwargs["model_path"] = model_path

    matcher = get_stereo_matcher(matcher_name=matcher_type, **extra_kwargs)
    raw_disparity, raw_valid_mask = matcher.compute_disparity(rect_main, rect_uw_norm)

    # 5. Apply Multi-Stage Disparity Post-Processing (LR check, speckle filtering, guided WLS, median)
    post_processor = DisparityPostProcessor()
    filtered_disparity, final_valid_mask = post_processor.process(
        disp_left=raw_disparity,
        guidance_img=rect_main,
        valid_mask=raw_valid_mask
    )

    # Quality Gate 3: Validate disparity fill rate
    if strict_quality_gates:
        validate_disparity(filtered_disparity)

    # 6. Convert disparity to physical metric depth (Z = f * B / d)
    f_rect = P1[0, 0]
    # Use the active T (refined calibration if available, not stale frame_data["T"])
    T_active = T.flatten() if hasattr(T, 'flatten') else np.array(T).flatten()
    baseline_norm = float(np.linalg.norm(T_active))
    baseline_m = baseline_norm / 1000.0 if baseline_norm > 1.0 else baseline_norm
    print(f"[Pipeline] Using baseline: {baseline_norm:.2f} mm ({baseline_m:.6f} m)")
    
    depth_calc = DepthCalculator(focal_length=f_rect, baseline=baseline_m, doffs=0.0)
    depth_map_est = depth_calc.disparity_to_depth(filtered_disparity)

    # Quality Gate 4: Validate depth plausibility
    if strict_quality_gates:
        validate_depth(depth_map_est)

    # 7. Point Cloud ICP Registration against RealSense Ground Truth (if available)
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

        # Evaluate iPhone depth error against RealSense ground truth
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
        rect_uw=rect_uw_norm,
        disparity_map=filtered_disparity,
        depth_map_m=depth_map_est,
        focal_length_px=float(f_rect),
        baseline_m=float(baseline_m),
        output_dir=output_dir,
        scene_name=os.path.basename(sequence_dir) or "iPhone Stereo Capture",
        eval_metrics=eval_metrics,
        raw_main=frame_data["img_main"],
        raw_uw=frame_data["img_uw"]
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
