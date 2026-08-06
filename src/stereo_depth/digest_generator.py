"""
Digest Generator script: Batch processes dataset scenes in data/ and generates HTML digest dashboard in digest/.
"""

import os
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.stereo_depth.pfm_io import read_pfm
from src.stereo_depth.calib import StereoCalibration
from src.stereo_depth.sliding_window import SlidingWindowMatcher
from src.stereo_depth.wls_filter import WLSDisparityFilter
from src.stereo_depth.depth_calculator import DepthCalculator
from src.stereo_depth.evaluator import DepthEvaluator


def apply_colormap(img_float, vmin=None, vmax=None, cmap_name='turbo'):
    """
    Normalizes a float array and applies a OpenCV colormap.
    """
    valid = np.isfinite(img_float) & (img_float > 0)
    if not np.any(valid):
        return np.zeros((img_float.shape[0], img_float.shape[1], 3), dtype=np.uint8)

    if vmin is None:
        vmin = np.percentile(img_float[valid], 2)
    if vmax is None:
        vmax = np.percentile(img_float[valid], 98)

    norm = np.clip((img_float - vmin) / max(vmax - vmin, 1e-5), 0, 1)
    norm_uint8 = (norm * 255).astype(np.uint8)

    # Use OpenCV colormap
    if cmap_name == 'turbo':
        cmap_code = cv2.COLORMAP_TURBO
    elif cmap_name == 'jet':
        cmap_code = cv2.COLORMAP_JET
    elif cmap_name == 'magma' or cmap_name == 'inferno':
        cmap_code = cv2.COLORMAP_INFERNO
    elif cmap_name == 'plasma':
        cmap_code = cv2.COLORMAP_PLASMA
    elif cmap_name == 'viridis':
        cmap_code = cv2.COLORMAP_VIRIDIS
    else:
        cmap_code = cv2.COLORMAP_TURBO

    colored = cv2.applyColorMap(norm_uint8, cmap_code)
    # Zero out invalid pixels to black
    colored[~valid] = [0, 0, 0]
    return colored


def process_scene(scene_dir, output_assets_dir, downsample_scale=0.5):
    scene_name = os.path.basename(scene_dir)
    print(f"Processing scene: {scene_name}...")

    calib_path = os.path.join(scene_dir, 'calib.txt')
    im0_path = os.path.join(scene_dir, 'im0.png')
    im1_path = os.path.join(scene_dir, 'im1.png')
    disp0_path = os.path.join(scene_dir, 'disp0.pfm')

    if not os.path.exists(im0_path) or not os.path.exists(im1_path):
        print(f"Skipping {scene_name}: missing im0.png or im1.png")
        return None

    calib = StereoCalibration.parse_file(calib_path)
    im0 = cv2.imread(im0_path)
    im1 = cv2.imread(im1_path)

    gt_disp = None
    if os.path.exists(disp0_path):
        gt_disp, _ = read_pfm(disp0_path)

    # Downsample for faster processing if needed while scaling max_disparity
    if downsample_scale != 1.0:
        h_new, w_new = int(im0.shape[0] * downsample_scale), int(im0.shape[1] * downsample_scale)
        im0_proc = cv2.resize(im0, (w_new, h_new))
        im1_proc = cv2.resize(im1, (w_new, h_new))
        ndisp_proc = max(16, int(calib.ndisp * downsample_scale))
        if gt_disp is not None:
            gt_disp_proc = cv2.resize(gt_disp, (w_new, h_new)) * downsample_scale
            # Scale inf values correctly
            gt_disp_proc[~np.isfinite(gt_disp_proc)] = np.inf
        else:
            gt_disp_proc = None
        f_proc = calib.focal_length * downsample_scale
    else:
        im0_proc, im1_proc = im0, im1
        ndisp_proc = calib.ndisp
        gt_disp_proc = gt_disp
        f_proc = calib.focal_length

    # 0. Raw Matching Cost Functions Benchmark (SAD vs SSD vs NCC vs ZNCC)
    matcher_sad = SlidingWindowMatcher(window_size=7, max_disparity=ndisp_proc, min_disparity=0, metric='sad')
    matcher_ssd = SlidingWindowMatcher(window_size=7, max_disparity=ndisp_proc, min_disparity=0, metric='ssd')
    matcher_ncc = SlidingWindowMatcher(window_size=7, max_disparity=ndisp_proc, min_disparity=0, metric='ncc')
    matcher_zncc = SlidingWindowMatcher(window_size=7, max_disparity=ndisp_proc, min_disparity=0, metric='zncc')

    t0_sad = time.time()
    disp_cost_sad = matcher_sad.compute_raw_disparity(im0_proc, im1_proc)
    lat_sad_ms = round((time.time() - t0_sad) * 1000.0, 2)

    t0_ssd = time.time()
    disp_cost_ssd = matcher_ssd.compute_raw_disparity(im0_proc, im1_proc)
    lat_ssd_ms = round((time.time() - t0_ssd) * 1000.0, 2)

    t0_ncc = time.time()
    disp_cost_ncc = matcher_ncc.compute_raw_disparity(im0_proc, im1_proc)
    lat_ncc_ms = round((time.time() - t0_ncc) * 1000.0, 2)

    t0_zncc = time.time()
    disp_cost_zncc = matcher_zncc.compute_raw_disparity(im0_proc, im1_proc)
    lat_zncc_ms = round((time.time() - t0_zncc) * 1000.0, 2)

    disp_step0_raw = disp_cost_sad
    latency_raw_ms = lat_sad_ms

    # Step 1: Raw SAD + Sub-Pixel Interpolation (N vs N-1: +Subpixel)
    disp_step1_subpixel, _ = matcher_sad.compute_disparity(im0_proc, im1_proc, enable_subpixel=True, check_lr_consistency=False)

    # Step 2: Raw SAD + Sub-Pixel + Left-Right Consistency Occlusion Check (N vs N-1: +Occlusion)
    disp_step2_occlusion, mask_step2 = matcher_sad.compute_disparity(im0_proc, im1_proc, enable_subpixel=True, check_lr_consistency=True)

    # Step 3: Mechanism A (NCC + Sub-Pixel + L-R Consistency Check + Median Filter)
    t0_a = time.time()
    disp_a_raw, mask_a = matcher_ncc.compute_disparity(im0_proc, im1_proc, enable_subpixel=True, check_lr_consistency=True)
    disp_step3_median = cv2.medianBlur(disp_a_raw.astype(np.float32), 5)
    latency_a_ms = round((time.time() - t0_a) * 1000.0, 2)

    # Step 4: Mechanism C (ZNCC + Sub-Pixel + WLS Edge-Preserving Regularization)
    t0_c = time.time()
    disp_c_raw, mask_c = matcher_zncc.compute_disparity(im0_proc, im1_proc, enable_subpixel=True, check_lr_consistency=False)
    wls = WLSDisparityFilter(lambda_val=8000.0, sigma_val=1.5)
    disp_step4_wls = wls.filter(disp_c_raw, im0_proc)
    latency_c_ms = round((time.time() - t0_c) * 1000.0, 2)

    # Aliases
    disp_raw = disp_step0_raw
    disp_a = disp_step3_median
    disp_c = disp_step4_wls

    # 3. Depth Calculation
    depth_calc = DepthCalculator(focal_length=f_proc, baseline=calib.baseline, doffs=calib.doffs * downsample_scale)
    depth_a = depth_calc.disparity_to_depth(disp_a)
    depth_c = depth_calc.disparity_to_depth(disp_c)

    # 4. Step-by-Step N vs N-1 Evaluations against Ground Truth
    eval_step0 = DepthEvaluator.evaluate_disparity(disp_step0_raw, gt_disp_proc) if gt_disp_proc is not None else {}
    eval_step1 = DepthEvaluator.evaluate_disparity(disp_step1_subpixel, gt_disp_proc) if gt_disp_proc is not None else {}
    eval_step2 = DepthEvaluator.evaluate_disparity(disp_step2_occlusion, gt_disp_proc) if gt_disp_proc is not None else {}
    eval_step3 = DepthEvaluator.evaluate_disparity(disp_step3_median, gt_disp_proc) if gt_disp_proc is not None else {}
    eval_step4 = DepthEvaluator.evaluate_disparity(disp_step4_wls, gt_disp_proc) if gt_disp_proc is not None else {}

    # N vs N-1 Delta Error Improvement (% RMSE reduction over previous step)
    rmse0 = eval_step0.get('rmse', 1.0)
    rmse1 = eval_step1.get('rmse', 1.0)
    rmse2 = eval_step2.get('rmse', 1.0)
    rmse3 = eval_step3.get('rmse', 1.0)
    rmse4 = eval_step4.get('rmse', 1.0)

    delta_step1_vs_0 = round(((rmse0 - rmse1) / max(rmse0, 1e-5)) * 100.0, 2)
    delta_step2_vs_1 = round(((rmse1 - rmse2) / max(rmse1, 1e-5)) * 100.0, 2)
    delta_step3_vs_2 = round(((rmse2 - rmse3) / max(rmse2, 1e-5)) * 100.0, 2)
    delta_step4_vs_3 = round(((rmse3 - rmse4) / max(rmse3, 1e-5)) * 100.0, 2)

    eval_raw = eval_step0
    eval_a = eval_step3
    eval_c = eval_step4
    comp_ac = DepthEvaluator.compare_pipelines(disp_a, disp_c)

    # 5. Difference Heatmap (|Disp_A - Disp_C|)
    diff_ac = np.abs(disp_a - disp_c)

    # 6. Grayscale Disparity Intensity Maps (Pixel intensity = Disparity value)
    v_max_disp = float(ndisp_proc)
    disp_raw_intensity = np.clip((disp_raw / v_max_disp) * 255.0, 0, 255).astype(np.uint8)
    disp_a_intensity = np.clip((disp_a / v_max_disp) * 255.0, 0, 255).astype(np.uint8)
    disp_c_intensity = np.clip((disp_c / v_max_disp) * 255.0, 0, 255).astype(np.uint8)

    # Compute luminance intensity statistics & 16-bin histograms
    hist_raw, _ = np.histogram(disp_raw_intensity, bins=16, range=(0, 256))
    hist_a, _ = np.histogram(disp_a_intensity, bins=16, range=(0, 256))
    hist_c, _ = np.histogram(disp_c_intensity, bins=16, range=(0, 256))

    stats_raw = {
        'min_intensity': float(np.min(disp_raw_intensity)),
        'max_intensity': float(np.max(disp_raw_intensity)),
        'mean_intensity': float(np.mean(disp_raw_intensity)),
        'histogram': hist_raw.tolist()
    }

    stats_a = {
        'min_intensity': float(np.min(disp_a_intensity)),
        'max_intensity': float(np.max(disp_a_intensity)),
        'mean_intensity': float(np.mean(disp_a_intensity)),
        'histogram': hist_a.tolist()
    }

    stats_c = {
        'min_intensity': float(np.min(disp_c_intensity)),
        'max_intensity': float(np.max(disp_c_intensity)),
        'mean_intensity': float(np.mean(disp_c_intensity)),
        'histogram': hist_c.tolist()
    }

    # Save Asset Images
    scene_asset_dir = os.path.join(output_assets_dir, scene_name)
    os.makedirs(scene_asset_dir, exist_ok=True)

    cv2.imwrite(os.path.join(scene_asset_dir, 'im0.jpg'), im0_proc)
    cv2.imwrite(os.path.join(scene_asset_dir, 'im1.jpg'), im1_proc)
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_raw.jpg'), apply_colormap(disp_raw, 0, v_max_disp, 'turbo'))
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_raw_intensity.jpg'), disp_raw_intensity)
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_a_intensity.jpg'), disp_a_intensity)
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_c_intensity.jpg'), disp_c_intensity)

    # 4-up Panel: Raw RGB, GT Intensity, Pipe A Intensity, Pipe C Intensity
    disp_a_bgr = cv2.cvtColor(disp_a_intensity, cv2.COLOR_GRAY2BGR)
    disp_c_bgr = cv2.cvtColor(disp_c_intensity, cv2.COLOR_GRAY2BGR)
    disp_raw_bgr = cv2.cvtColor(disp_raw_intensity, cv2.COLOR_GRAY2BGR)

    if gt_disp_proc is not None:
        gt_intensity = np.clip((gt_disp_proc / v_max_disp) * 255.0, 0, 255).astype(np.uint8)
        gt_intensity[~np.isfinite(gt_disp_proc)] = 0
        gt_bgr = cv2.cvtColor(gt_intensity, cv2.COLOR_GRAY2BGR)
        hist_gt, _ = np.histogram(gt_intensity[np.isfinite(gt_disp_proc)], bins=16, range=(0, 256))
        stats_gt = {
            'min_intensity': float(np.min(gt_intensity)),
            'max_intensity': float(np.max(gt_intensity)),
            'mean_intensity': float(np.mean(gt_intensity)),
            'histogram': hist_gt.tolist()
        }
        cv2.imwrite(os.path.join(scene_asset_dir, 'disp_gt_intensity.jpg'), gt_intensity)
        cv2.imwrite(os.path.join(scene_asset_dir, 'disp_gt.jpg'), apply_colormap(gt_disp_proc, 0, v_max_disp, 'turbo'))
    else:
        gt_bgr = np.zeros_like(im0_proc)
        stats_gt = {}

    top_row = np.hstack((im0_proc, gt_bgr))
    bot_row = np.hstack((disp_raw_bgr, disp_c_bgr))
    panel_4up = np.vstack((top_row, bot_row))
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_intensity_panel.jpg'), panel_4up)

    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_a.jpg'), apply_colormap(disp_a, 0, v_max_disp, 'turbo'))
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_c.jpg'), apply_colormap(disp_c, 0, v_max_disp, 'turbo'))
    cv2.imwrite(os.path.join(scene_asset_dir, 'diff_ac.jpg'), apply_colormap(diff_ac, 0, np.percentile(diff_ac, 98), 'inferno'))

    # Scanline Depth Profile (mid-height scanline)
    mid_y = im0_proc.shape[0] // 2
    profile_gt = gt_disp_proc[mid_y, :].tolist() if gt_disp_proc is not None else []
    profile_raw = disp_raw[mid_y, :].tolist()
    profile_a = disp_a[mid_y, :].tolist()
    profile_c = disp_c[mid_y, :].tolist()

    # Cost Functions Evaluation & Image Exports
    eval_sad = DepthEvaluator.evaluate_disparity(disp_cost_sad, gt_disp_proc) if gt_disp_proc is not None else {}
    eval_ssd = DepthEvaluator.evaluate_disparity(disp_cost_ssd, gt_disp_proc) if gt_disp_proc is not None else {}
    eval_ncc = DepthEvaluator.evaluate_disparity(disp_cost_ncc, gt_disp_proc) if gt_disp_proc is not None else {}
    eval_zncc = DepthEvaluator.evaluate_disparity(disp_cost_zncc, gt_disp_proc) if gt_disp_proc is not None else {}

    disp_sad_int = np.clip((disp_cost_sad / v_max_disp) * 255.0, 0, 255).astype(np.uint8)
    disp_ssd_int = np.clip((disp_cost_ssd / v_max_disp) * 255.0, 0, 255).astype(np.uint8)
    disp_ncc_int = np.clip((disp_cost_ncc / v_max_disp) * 255.0, 0, 255).astype(np.uint8)
    disp_zncc_int = np.clip((disp_cost_zncc / v_max_disp) * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_cost_sad_intensity.jpg'), disp_sad_int)
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_cost_ssd_intensity.jpg'), disp_ssd_int)
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_cost_ncc_intensity.jpg'), disp_ncc_int)
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_cost_zncc_intensity.jpg'), disp_zncc_int)

    # 4-up Cost Functions Panel (SAD, SSD, NCC, ZNCC)
    top_cost_row = np.hstack((cv2.cvtColor(disp_sad_int, cv2.COLOR_GRAY2BGR), cv2.cvtColor(disp_ssd_int, cv2.COLOR_GRAY2BGR)))
    bot_cost_row = np.hstack((cv2.cvtColor(disp_ncc_int, cv2.COLOR_GRAY2BGR), cv2.cvtColor(disp_zncc_int, cv2.COLOR_GRAY2BGR)))
    cost_panel_4up = np.vstack((top_cost_row, bot_cost_row))
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_cost_functions_panel.jpg'), cost_panel_4up)

    # 3D Point Cloud sampling (Pipeline C)
    pts, colors = depth_calc.generate_point_cloud(depth_c, im0_proc, subsample_step=6, max_points=3000)

    # Export step-by-step N vs N-1 Intensity Maps
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_step0_raw_intensity.jpg'), disp_raw_intensity)
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_step1_subpixel_intensity.jpg'), np.clip((disp_step1_subpixel / v_max_disp) * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(scene_asset_dir, 'disp_step2_occlusion_intensity.jpg'), np.clip((disp_step2_occlusion / v_max_disp) * 255.0, 0, 255).astype(np.uint8))

    scene_data = {
        'name': scene_name,
        'width': int(im0_proc.shape[1]),
        'height': int(im0_proc.shape[0]),
        'baseline': float(calib.baseline),
        'focal_length': float(f_proc),
        'ndisp': int(ndisp_proc),
        'latency_raw_ms': float(latency_raw_ms),
        'latency_a_ms': float(latency_a_ms),
        'latency_c_ms': float(latency_c_ms),
        'stats_raw': stats_raw,
        'stats_a': stats_a,
        'stats_c': stats_c,
        'stats_gt': stats_gt,
        'eval_cost_functions': {
            'sad': eval_sad,
            'ssd': eval_ssd,
            'ncc': eval_ncc,
            'zncc': eval_zncc
        },
        'latency_cost_functions_ms': {
            'sad': lat_sad_ms,
            'ssd': lat_ssd_ms,
            'ncc': lat_ncc_ms,
            'zncc': lat_zncc_ms
        },
        'eval_step0_raw': eval_step0,
        'eval_step1_subpixel': eval_step1,
        'eval_step2_occlusion': eval_step2,
        'eval_step3_median': eval_step3,
        'eval_step4_wls': eval_step4,
        'deltas': {
            'step1_vs_0_pct': delta_step1_vs_0,
            'step2_vs_1_pct': delta_step2_vs_1,
            'step3_vs_2_pct': delta_step3_vs_2,
            'step4_vs_3_pct': delta_step4_vs_3
        },
        'eval_raw': eval_raw,
        'eval_a': eval_a,
        'eval_c': eval_c,
        'comp_ac': comp_ac,
        'mid_scanline_y': int(mid_y),
        'scanline_gt': [float(x) if np.isfinite(x) else 0.0 for x in profile_gt],
        'scanline_raw': [float(x) for x in profile_raw],
        'scanline_step1': [float(x) for x in disp_step1_subpixel[mid_y, :]],
        'scanline_step2': [float(x) for x in disp_step2_occlusion[mid_y, :]],
        'scanline_a': [float(x) for x in profile_a],
        'scanline_c': [float(x) for x in profile_c],
        'point_cloud': {
            'points': pts.astype(float).tolist(),
            'colors': colors.astype(int).tolist()
        }
    }

    print(f"Scene {scene_name} completed. Latency: A={latency_a_ms}ms, C={latency_c_ms}ms. RMSE: A={eval_a.get('rmse', 'N/A')}, C={eval_c.get('rmse', 'N/A')}")
    return scene_data


def generate_digest_data(data_dir='data', output_dir='digest'):
    os.makedirs(output_dir, exist_ok=True)
    assets_dir = os.path.join(output_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    scene_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    scene_dirs.sort()

    all_scenes_data = []
    # Process up to 8 representative dataset scenes for optimal web speed
    for s_dir in scene_dirs[:8]:
        s_data = process_scene(s_dir, assets_dir, downsample_scale=0.35)
        if s_data:
            all_scenes_data.append(s_data)

    data_json_path = os.path.join(output_dir, 'data.json')
    with open(data_json_path, 'w') as f:
        json.dump({'scenes': all_scenes_data}, f, indent=2)

    print(f"Saved dataset JSON to {data_json_path}")
    return all_scenes_data
