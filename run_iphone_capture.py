"""
CLI tool for running the Real-World Depth Pipeline on custom iPhone stereo photos (Main + Ultra-Wide).
Usage:
    python3 run_iphone_capture.py --main data/my_iphone/main.jpg --ultrawide data/my_iphone/ultrawide.jpg
"""

import argparse
import cv2

from src.realworld.stereo_artifacts import write_stereo_result
from src.realworld.stereo_contracts import (
    MatcherMethod,
    PixelSize,
    load_calibration_result,
)
from src.realworld.iphone_stereo_processor import IPhoneStereoProcessor
from serve_digest import serve_digest


def run_custom_iphone_depth(
    main_path: str,
    ultrawide_path: str,
    output_dir: str = "digest_iphone",
    target_size: tuple = None,
    calibration_result_path: str = "calibration-runs/current/result.json",
    matcher_method: str = "sliding_window",
):
    """
    Runs the shared limitation-aware processing core on an iPhone Stereo Pair.

    The processing path accepts only a versioned Calibration Result so units,
    transform direction, limitations, and trust remain explicit.
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

    print(f"[Calibration] Loading explicit Calibration Result: {calibration_result_path}")
    calibration_result = load_calibration_result(calibration_result_path)
    method = MatcherMethod(matcher_method)
    processor = IPhoneStereoProcessor(
        calibration_result=calibration_result,
        output_size=(
            PixelSize(int(target_size[0]), int(target_size[1]))
            if target_size is not None
            else None
        ),
    )
    result = processor.process(img_main, img_uw, method=method)
    write_stereo_result(
        result,
        output_dir,
        main_input=main_path,
        ultrawide_input=ultrawide_path,
    )

    print(f"\nStereo Processing Result created for the custom iPhone Stereo Pair.")
    print(f"Output files saved to directory: '{output_dir}/'")
    print(f"  - Interactive Dashboard: {output_dir}/index.html")
    print(f"  - Result disposition:    {result.disposition.value}")
    print(f"  - Trusted depth:         {result.trusted_depth_eligible}")
    print(f"  - Result manifest:       {output_dir}/stereo_result.json")
    if result.disposition.value != "rejected":
        print(f"  - Rectified UltraWide:   {output_dir}/rectified_ultrawide.png (physical left)")
        print(f"  - Rectified Main RGB:    {output_dir}/rectified_main.png (physical right)")
        print(f"  - Disparity Map:         {output_dir}/disparity_map.png")
        print(f"  - Metric Depth Map:      {output_dir}/depth_map.png")
        print(f"  - Depth Overlay:         {output_dir}/depth_overlay.png")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stereo depth pipeline on custom iPhone photos")
    parser.add_argument("--main", required=True, help="Path to Main camera photo (24mm/26mm Wide)")
    parser.add_argument("--ultrawide", required=True, help="Path to Ultra-Wide camera photo (13mm/14mm)")
    parser.add_argument("--calibration-result", default="calibration-runs/current/result.json", help="Explicit trusted or input-limited Calibration Result")
    parser.add_argument("--output", default="digest_iphone", help="Output directory for depth results")
    parser.add_argument("--no-view", action="store_true", help="Do not automatically launch web browser visualization")
    args = parser.parse_args()

    out_dir = run_custom_iphone_depth(
        main_path=args.main,
        ultrawide_path=args.ultrawide,
        output_dir=args.output,
        calibration_result_path=args.calibration_result,
    )
    if not args.no_view:
        serve_digest(out_dir, port=8080, open_browser=True, block=True)
