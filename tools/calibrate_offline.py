"""
Standalone Offline Stereo Calibration CLI Tool for iPhone Dual-Camera & Heterogeneous Pairs.
Processes folders of calibration target snapshots and exports refined intrinsic & extrinsic JSON matrices.
Includes pattern auto-discovery and iterative outlier pruning.
"""

import os
import sys
import json
import glob
import argparse
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.realworld.calibration_refiner import CalibrationRefiner


def load_image_pairs(images_dir: str):
    """
    Discovers matching (main, ultrawide) image pairs from a calibration directory.
    """
    pairs = []
    orig_uw_sizes = []
    
    # 1. Search for subdirectories inside images_dir (e.g. pair_01/main.jpg & pair_01/ultrawide.jpg)
    subdirs = sorted(glob.glob(os.path.join(images_dir, "*")))
    for sd in subdirs:
        if os.path.isdir(sd):
            main_path = os.path.join(sd, "main.jpg")
            uw_path = os.path.join(sd, "ultrawide.jpg")
            if not os.path.exists(main_path):
                main_path = os.path.join(sd, "im0.jpg")
                uw_path = os.path.join(sd, "im1.jpg")

            if os.path.exists(main_path) and os.path.exists(uw_path):
                img_m = cv2.imread(main_path)
                img_u = cv2.imread(uw_path)
                if img_m is not None and img_u is not None:
                    orig_uw_sizes.append(img_u.shape[:2])
                    if img_m.shape[:2] != img_u.shape[:2]:
                        img_u = cv2.resize(img_u, (img_m.shape[1], img_m.shape[0]), interpolation=cv2.INTER_CUBIC)
                    pairs.append((img_m, img_u))

    # 2. Search for direct files matching pattern frame_*_main.jpg / frame_*_ultrawide.jpg
    if len(pairs) == 0:
        main_files = sorted(glob.glob(os.path.join(images_dir, "*main*.jpg"))) + sorted(glob.glob(os.path.join(images_dir, "*main*.png")))
        for mf in main_files:
            uf = mf.replace("main", "ultrawide")
            if not os.path.exists(uf):
                uf = mf.replace("main", "uw")
            if os.path.exists(uf):
                img_m = cv2.imread(mf)
                img_u = cv2.imread(uf)
                if img_m is not None and img_u is not None:
                    orig_uw_sizes.append(img_u.shape[:2])
                    if img_m.shape[:2] != img_u.shape[:2]:
                        img_u = cv2.resize(img_u, (img_m.shape[1], img_m.shape[0]), interpolation=cv2.INTER_CUBIC)
                    pairs.append((img_m, img_u))

    return pairs, orig_uw_sizes


def generate_rectification_preview(img_m: np.ndarray, img_u: np.ndarray, K1: np.ndarray, D1: np.ndarray, K2: np.ndarray, D2: np.ndarray, R: np.ndarray, T: np.ndarray, save_path: str):
    """
    Computes stereo rectification and saves a horizontal epipolar line visual verification image.
    """
    h, w = img_m.shape[:2]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    rect_m = cv2.remap(img_m, map1_x, map1_y, cv2.INTER_LANCZOS4)
    rect_u = cv2.remap(img_u, map2_x, map2_y, cv2.INTER_LANCZOS4)

    # Draw horizontal epipolar grid lines every 30 pixels
    combo = np.hstack((rect_m, rect_u))
    for y in range(0, h, 30):
        cv2.line(combo, (0, y), (w * 2, y), (0, 255, 0), 1)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    cv2.imwrite(save_path, combo)


def main():
    parser = argparse.ArgumentParser(description="Standalone Offline Stereo Calibration CLI Tool")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing calibration snapshot image pairs")
    parser.add_argument("--pattern_cols", type=int, default=5, help="Number of inner corners along width")
    parser.add_argument("--pattern_rows", type=int, default=5, help="Number of inner corners along height")
    parser.add_argument("--square_size_m", type=float, default=0.026, help="Square size in meters (default: 0.026m = 26mm)")
    parser.add_argument("--auto_discover", action="store_true", help="Automatically discover checkerboard grid dimensions")
    parser.add_argument("--output_json", type=str, default="calibration_refined.json", help="Path to save refined calibration JSON")
    parser.add_argument("--preview_image", type=str, default="calib_rectified_verification.jpg", help="Path to save epipolar line verification image")

    args = parser.parse_args()

    print(f"=== Running Offline Stereo Calibration on '{args.images_dir}' ===")
    pairs, orig_uw_sizes = load_image_pairs(args.images_dir)
    print(f"[CalibCLI] Discovered {len(pairs)} calibration snapshot pairs.")

    if len(pairs) == 0:
        print("[CalibCLI] Error: No valid image pairs found in directory.")
        sys.exit(1)

    h, w = pairs[0][0].shape[:2]
    f_est = max(w, h) * 0.8
    initial_K1 = np.array([[f_est, 0.0, w / 2.0], [0.0, f_est, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    initial_K2 = np.array([[f_est * 0.5, 0.0, w / 2.0], [0.0, f_est * 0.5, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    refiner = CalibrationRefiner(
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size_m=args.square_size_m,
        auto_discover=args.auto_discover
    )
    calib_results = refiner.calibrate_stereo(pairs, initial_K1, initial_K2)

    K2_calibrated = calib_results["K2"].copy()

    main_h, main_w = pairs[0][0].shape[:2]
    orig_uw_h, orig_uw_w = orig_uw_sizes[0] if orig_uw_sizes else (main_h, main_w)

    scale_x = orig_uw_w / main_w
    scale_y = orig_uw_h / main_h
    K2_export = calib_results["K2"].copy()
    K2_export[0, 0] *= scale_x  # fx
    K2_export[1, 1] *= scale_y  # fy
    K2_export[0, 2] *= scale_x  # cx
    K2_export[1, 2] *= scale_y  # cy
    calib_results["K2"] = K2_export

    export_data = {
        "K1": calib_results["K1"].tolist(),
        "D1": calib_results["D1"].tolist(),
        "K2": calib_results["K2"].tolist(),
        "D2": calib_results["D2"].tolist(),
        "R": calib_results["R"].tolist(),
        "T": calib_results["T"].tolist(),
        "reprojection_error": calib_results["reprojection_error"],
        "num_pairs_used": calib_results["num_pairs_used"],
        "pattern_size": list(calib_results.get("pattern_size", (args.pattern_cols, args.pattern_rows))),
        "image_size_main": [main_w, main_h],
        "image_size_ultrawide": [orig_uw_w, orig_uw_h]
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(export_data, f, indent=2)

    num_used = calib_results["num_pairs_used"]
    if num_used > 0:
        # Generate epipolar grid verification preview using first inlier pair
        inlier_idx = calib_results["inlier_indices"][0] if calib_results.get("inlier_indices") else 0
        img_m, img_u = pairs[inlier_idx]
        generate_rectification_preview(
            img_m, img_u,
            calib_results["K1"], calib_results["D1"],
            K2_calibrated, calib_results["D2"],
            calib_results["R"], calib_results["T"],
            args.preview_image
        )

        baseline_mm = abs(float(calib_results["T"][0][0])) * 1000.0 if len(calib_results["T"]) > 0 else 0.0

        print(f"\n========================================================")
        print(f"🎉 STEREO CALIBRATION SOLVED SUCCESSFULLY")
        print(f"========================================================")
        print(f"  - Pattern Grid Size:   {export_data['pattern_size'][0]} cols x {export_data['pattern_size'][1]} rows")
        print(f"  - Inlier Pairs Used:   {num_used} / {len(pairs)}")
        print(f"  - Reprojection Error:  {calib_results['reprojection_error']:.4f} px")
        print(f"  - Solved Baseline T_x: {baseline_mm:.2f} mm")
        print(f"  - Refined Output JSON: {args.output_json}")
        print(f"  - Epipolar Preview:    {args.preview_image}")
        print(f"========================================================\n")
    else:
        print(f"\n========================================================")
        print(f"⚠️ STEREO CALIBRATION FAILED — 0 PAIRS MATCHED")
        print(f"========================================================")
        print(f"  - Discovered Snapshot Pairs: {len(pairs)}")
        print(f"  - Valid Pattern Matches:     0 / {len(pairs)}")
        print(f"  - Fallback Output JSON:      {args.output_json}")
        print(f"--------------------------------------------------------")
        print(f"Troubleshooting Instructions:")
        print(f"  1. Check --pattern_cols and --pattern_rows:")
        print(f"     Must be INNER CORNER COUNT, not square count.")
        print(f"  2. Ensure the board is clearly visible in BOTH Main and Ultra-Wide views.")
        print(f"  3. Move closer to the target so the board covers at least 20% of the frame.")
        print(f"========================================================\n")


if __name__ == "__main__":
    main()
