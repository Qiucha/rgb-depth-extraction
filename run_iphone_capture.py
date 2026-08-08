"""
CLI tool for running the Real-World Depth Pipeline on custom iPhone stereo photos (Main + Ultra-Wide).
Usage:
    python3 run_iphone_capture.py --main data/my_iphone/main.jpg --ultrawide data/my_iphone/ultrawide.jpg
"""

import os
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from src.realworld.hetero_rectifier import HeterogeneousStereoRectifier
from src.realworld.digest_builder import generate_realworld_digest
from src.stereo_depth.sliding_window import SlidingWindowMatcher
from src.stereo_depth.wls_filter import WLSDisparityFilter
from src.stereo_depth.depth_calculator import DepthCalculator
from serve_digest import serve_digest


def extract_exif_focal_length(image_path: str):
    """
    Extracts focal length and image dimensions from EXIF metadata if available.
    """
    try:
        img = Image.open(image_path)
        w, h = img.size
        exif_data = img._getexif()
        focal_mm = None
        focal_35mm = None

        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == 'FocalLength':
                    focal_mm = float(value)
                elif tag_name == 'FocalLengthIn35mmFilm':
                    focal_35mm = float(value)

        return w, h, focal_mm, focal_35mm
    except Exception:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        return w, h, None, None


def run_custom_iphone_depth(main_path: str, ultrawide_path: str, output_dir: str = "digest_iphone", target_size: tuple = (1280, 960)):
    """
    Runs realworld depth extraction pipeline on custom user-provided iPhone photos.
    """
    print(f"=== Running Custom iPhone Stereo Depth Extraction ===")
    print(f"Main Camera Image: {main_path}")
    print(f"Ultra-Wide Image:  {ultrawide_path}")

    # 1. Load images
    img_main = cv2.imread(main_path)
    img_uw = cv2.imread(ultrawide_path)

    if img_main is None:
        raise FileNotFoundError(f"Could not read Main image at {main_path}")
    if img_uw is None:
        raise FileNotFoundError(f"Could not read Ultra-Wide image at {ultrawide_path}")

    w1, h1, f_mm1, f35_1 = extract_exif_focal_length(main_path)
    w2, h2, f_mm2, f35_2 = extract_exif_focal_length(ultrawide_path)

    # Estimate intrinsic matrices
    # Equivalent 35mm focal lengths: Main ~ 24-26mm, Ultra-Wide ~ 13-14mm
    # Sensor 35mm width = 36mm
    f1_px = (f35_1 / 36.0 * w1) if f35_1 else (0.75 * w1)
    f2_px = (f35_2 / 36.0 * w2) if f35_2 else (0.36 * w2)

    K1 = np.array([
        [f1_px, 0.0, w1 / 2.0],
        [0.0, f1_px, h1 / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    K2 = np.array([
        [f2_px, 0.0, w2 / 2.0],
        [0.0, f2_px, h2 / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Nominal iPhone baseline T = 19.5mm
    R = np.eye(3, dtype=np.float64)
    T = np.array([19.5, 0.0, 0.0], dtype=np.float64)

    # 2. Perform FOV alignment and single-pass dynamic rectification
    rectifier = HeterogeneousStereoRectifier(target_size=target_size)
    rect_main, rect_uw, P1, P2, Q = rectifier.rectify_pair(
        img_main=img_main, img_uw=img_uw,
        K1=K1, D1=np.zeros(5),
        K2=K2, D2=np.zeros(5),
        R=R, T=T
    )

    gray_main = cv2.cvtColor(rect_main, cv2.COLOR_BGR2GRAY)
    gray_uw = cv2.cvtColor(rect_uw, cv2.COLOR_BGR2GRAY)

    # 3. Stereo Matching & WLS Filtering
    matcher = SlidingWindowMatcher(window_size=7, max_disparity=64, metric="zncc")
    raw_disparity, valid_mask = matcher.compute_disparity(gray_main, gray_uw)

    wls = WLSDisparityFilter(lambda_val=8000.0, sigma_val=1.5)
    filtered_disparity = wls.filter(raw_disparity, gray_main)

    # 4. Metric Depth Calculation
    f_rect = P1[0, 0]
    baseline_m = 0.0195  # 19.5 mm
    depth_calc = DepthCalculator(focal_length=f_rect, baseline=baseline_m, doffs=0.0)
    depth_map_m = depth_calc.disparity_to_depth(filtered_disparity)

    # 5. Output Visualizations
    os.makedirs(output_dir, exist_ok=True)

    # Colorize disparity
    disp_norm = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    # Colorize depth (0.2m to 3.0m)
    depth_clamped = np.clip(depth_map_m, 0.2, 3.0)
    depth_norm = cv2.normalize(depth_clamped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_PLASMA)

    # Depth overlay on Main RGB
    overlay = cv2.addWeighted(rect_main, 0.6, depth_color, 0.4, 0)

    cv2.imwrite(os.path.join(output_dir, "rectified_main.png"), rect_main)
    cv2.imwrite(os.path.join(output_dir, "rectified_ultrawide.png"), rect_uw)
    cv2.imwrite(os.path.join(output_dir, "disparity_map.png"), disp_color)
    cv2.imwrite(os.path.join(output_dir, "depth_map.png"), depth_color)
    cv2.imwrite(os.path.join(output_dir, "depth_overlay.png"), overlay)

    # Generate Visual Digest Telemetry & Interactive HTML Dashboard Studio
    generate_realworld_digest(
        rect_main=rect_main,
        rect_uw=rect_uw,
        disparity_map=filtered_disparity,
        depth_map_m=depth_map_m,
        focal_length_px=float(f_rect),
        baseline_m=float(baseline_m),
        output_dir=output_dir,
        scene_name="Custom iPhone Stereo Photo Pair"
    )

    # Write summary metadata
    summary = {
        "main_input": main_path,
        "ultrawide_input": ultrawide_path,
        "rectified_resolution": target_size,
        "rectified_focal_length_px": float(f_rect),
        "baseline_meters": baseline_m,
        "min_depth_m": float(np.min(depth_map_m[depth_map_m > 0.1])),
        "max_depth_m": float(np.max(depth_map_m[depth_map_m < 10.0]))
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSuccessfully processed custom iPhone stereo pair!")
    print(f"Output files saved to directory: '{output_dir}/'")
    print(f"  - Interactive Dashboard: {output_dir}/index.html")
    print(f"  - Rectified Main RGB:     {output_dir}/rectified_main.png")
    print(f"  - Rectified UltraWide:   {output_dir}/rectified_ultrawide.png")
    print(f"  - Disparity Map:         {output_dir}/disparity_map.png")
    print(f"  - Metric Depth Map:      {output_dir}/depth_map.png")
    print(f"  - Depth Overlay:         {output_dir}/depth_overlay.png")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stereo depth pipeline on custom iPhone photos")
    parser.add_argument("--main", required=True, help="Path to Main camera photo (24mm/26mm Wide)")
    parser.add_argument("--ultrawide", required=True, help="Path to Ultra-Wide camera photo (13mm/14mm)")
    parser.add_argument("--output", default="digest_iphone", help="Output directory for depth results")
    parser.add_argument("--no-view", action="store_true", help="Do not automatically launch web browser visualization")
    args = parser.parse_args()

    out_dir = run_custom_iphone_depth(args.main, args.ultrawide, args.output)
    if not args.no_view:
        serve_digest(out_dir, port=8080, open_browser=True, block=True)

