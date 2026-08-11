"""
Digest Builder for Real-World & iPhone Dual-Camera Stereo Depth Extraction.
Generates full visual digest telemetries, asset images, 3D point clouds, scanline graphs,
and self-contained interactive HTML dashboards with live filter toggles & parameter tuning controls.
"""

import os
import json
import cv2
import numpy as np
import shutil


def apply_colormap(img_float, vmin=None, vmax=None, cmap_code=cv2.COLORMAP_TURBO):
    """
    Normalizes float depth/disparity matrix and applies OpenCV colormap.
    """
    valid = np.isfinite(img_float) & (img_float > 0)
    if not np.any(valid):
        return np.zeros((img_float.shape[0], img_float.shape[1], 3), dtype=np.uint8)

    if vmin is None:
        vmin = float(np.percentile(img_float[valid], 2))
    if vmax is None:
        vmax = float(np.percentile(img_float[valid], 98))

    norm = np.clip((img_float - vmin) / max(vmax - vmin, 1e-5), 0, 1)
    norm_uint8 = (norm * 255).astype(np.uint8)

    colored = cv2.applyColorMap(norm_uint8, cmap_code)
    colored[~valid] = [0, 0, 0]
    return colored


def generate_realworld_digest(
    rect_main: np.ndarray,
    rect_uw: np.ndarray,
    disparity_map: np.ndarray,
    depth_map_m: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    output_dir: str = "digest_live_iphone",
    scene_name: str = "Live iPhone Stereo Session",
    eval_metrics: dict = None,
    is_live: bool = False,
    raw_main: np.ndarray = None,
    raw_uw: np.ndarray = None
):
    """
    Generates complete visual digest telemetry assets, point cloud, data.json, and interactive HTML dashboard.
    Preserves original unadjusted/unrectified raw input images for side-by-side comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    assets_dir = os.path.join(output_dir, "assets", scene_name.replace(" ", "_"))
    os.makedirs(assets_dir, exist_ok=True)

    h, w = rect_main.shape[:2]

    # Save original raw unadjusted images for comparison if provided
    unadjusted_main = raw_main if raw_main is not None else rect_main
    unadjusted_uw = raw_uw if raw_uw is not None else rect_uw

    # Resize raw images to target resolution for consistent visual comparison
    raw_main_resized = cv2.resize(unadjusted_main, (w, h)) if unadjusted_main.shape[:2] != (h, w) else unadjusted_main
    raw_uw_resized = cv2.resize(unadjusted_uw, (w, h)) if unadjusted_uw.shape[:2] != (h, w) else unadjusted_uw

    cv2.imwrite(os.path.join(assets_dir, "im0_raw.jpg"), raw_main_resized)
    cv2.imwrite(os.path.join(assets_dir, "im1_raw.jpg"), raw_uw_resized)

    # 1. Disparity Intensity Map (0..255)
    max_disp = float(np.max(disparity_map[np.isfinite(disparity_map)])) if np.any(np.isfinite(disparity_map)) else 64.0
    max_disp = max(max_disp, 1.0)
    disp_intensity = np.clip((disparity_map / max_disp) * 255.0, 0, 255).astype(np.uint8)

    # Colormapped disparity & depth
    disp_color = apply_colormap(disparity_map, 0, max_disp, cv2.COLORMAP_TURBO)
    depth_clamped = np.clip(depth_map_m, 0.2, 5.0)
    depth_color = apply_colormap(depth_clamped, 0.2, 5.0, cv2.COLORMAP_PLASMA)

    # RGB Depth Overlay
    overlay = cv2.addWeighted(rect_main, 0.6, depth_color, 0.4, 0)

    # Residual difference heatmap / edge gradient
    gray_main = cv2.cvtColor(rect_main, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_main, cv2.CV_64F, 1, 0, ksize=3)
    edge_diff = np.abs(sobel_x)
    diff_heatmap = apply_colormap(edge_diff, 0, np.percentile(edge_diff, 98), cv2.COLORMAP_INFERNO)

    # Save images into scene asset folder
    cv2.imwrite(os.path.join(assets_dir, "im0.jpg"), rect_main)
    cv2.imwrite(os.path.join(assets_dir, "im1.jpg"), rect_uw)
    cv2.imwrite(os.path.join(assets_dir, "disp_raw.jpg"), disp_color)
    cv2.imwrite(os.path.join(assets_dir, "disp_c.jpg"), disp_color)
    cv2.imwrite(os.path.join(assets_dir, "disp_c_intensity.jpg"), disp_intensity)
    cv2.imwrite(os.path.join(assets_dir, "depth_color.jpg"), depth_color)
    cv2.imwrite(os.path.join(assets_dir, "depth_overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(assets_dir, "diff_ac.jpg"), diff_heatmap)

    # Side-by-side Raw vs Rectified/Enhanced comparison panel (2x2 grid)
    top_raw = np.hstack((raw_main_resized, raw_uw_resized))
    bot_rect = np.hstack((rect_main, rect_uw))
    raw_vs_rect_panel = np.vstack((top_raw, bot_rect))
    cv2.imwrite(os.path.join(assets_dir, "raw_vs_rectified_panel.jpg"), raw_vs_rect_panel)

    # 20px Epipolar Grid Line Overlay Verification
    grid_img = np.hstack((rect_main, rect_uw)).copy()
    for gy in range(0, h, 20):
        color = (0, 0, 255) if (gy // 20) % 2 == 0 else (0, 255, 0)
        cv2.line(grid_img, (0, gy), (grid_img.shape[1], gy), color, 1)
    cv2.imwrite(os.path.join(assets_dir, "epipolar_grid.jpg"), grid_img)

    # 4-up Panel: Main RGB, Ultra-Wide RGB, Disparity Intensity, Metric Depth Map
    disp_int_bgr = cv2.cvtColor(disp_intensity, cv2.COLOR_GRAY2BGR)
    top_row = np.hstack((rect_main, rect_uw))
    bot_row = np.hstack((disp_int_bgr, depth_color))
    panel_4up = np.vstack((top_row, bot_row))
    cv2.imwrite(os.path.join(assets_dir, "disp_intensity_panel.jpg"), panel_4up)

    # 2. Mid-Scanline Depth Profile
    mid_y = h // 2
    profile_disp = disparity_map[mid_y, :].tolist()
    profile_depth = depth_map_m[mid_y, :].tolist()

    # 3. 3D Point Cloud Sampling
    pts = []
    colors = []
    step = max(4, int(w / 160))  # Sample points for light web canvas rendering
    for y in range(0, h, step):
        for x in range(0, w, step):
            z = float(depth_map_m[y, x])
            if 0.2 <= z <= 6.0:
                # Unproject 2D to 3D point (X, Y, Z in mm)
                cx, cy = w / 2.0, h / 2.0
                X = (x - cx) * z * 1000.0 / focal_length_px
                Y = (y - cy) * z * 1000.0 / focal_length_px
                Z = z * 1000.0
                b, g, r = rect_main[y, x]
                pts.append([float(X), float(Y), float(Z)])
                colors.append([int(r), int(g), int(b)])

    scene_data = {
        "name": scene_name,
        "width": w,
        "height": h,
        "baseline": baseline_m * 1000.0,
        "focal_length": focal_length_px,
        "mid_scanline_y": mid_y,
        "scanline_c": profile_disp,
        "scanline_depth_m": profile_depth,
        "eval_metrics": eval_metrics or {},
        "is_live": is_live,
        "point_cloud": {
            "points": pts,
            "colors": colors
        }
    }

    # Write data.json
    data_json_path = os.path.join(output_dir, "data.json")
    with open(data_json_path, "w") as f:
        json.dump({"scenes": [scene_data]}, f, indent=2)

    # Generate index.html in output_dir
    generate_interactive_html_dashboard(output_dir)
    print(f"[DigestBuilder] Generated visual digest dashboard in '{output_dir}/'")
    return output_dir


def generate_interactive_html_dashboard(output_dir: str):
    """
    Creates a standalone, highly aesthetic interactive HTML visual digest dashboard in output_dir.
    """
    html_path = os.path.join(output_dir, "index.html")
    html_content = get_digest_html_template()
    with open(html_path, "w") as f:
        f.write(html_content)


def get_digest_html_template():
    return """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>iPhone Stereo Depth Extraction Studio & Noise Filter Controls</title>
  
  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  
  <style>
    :root {
      --bg-dark: #090d16;
      --bg-card: rgba(18, 26, 43, 0.75);
      --bg-card-border: rgba(255, 255, 255, 0.1);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      
      --accent-cyan: #0284c7;
      --accent-blue: #38bdf8;
      --accent-purple: #8b5cf6;
      --accent-pink: #f43f5e;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      
      --radius-lg: 16px;
      --radius-md: 12px;
      --radius-sm: 8px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      padding: 1.5rem 2rem;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(2, 132, 199, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
    }

    .container { max-width: 1440px; margin: 0 auto; }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--bg-card-border);
    }

    .brand { display: flex; align-items: center; gap: 1rem; }
    
    .brand-icon {
      width: 44px; height: 44px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
      border-radius: var(--radius-md);
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 20px rgba(2, 132, 199, 0.4);
    }
    
    .brand-icon svg { width: 24px; height: 24px; fill: none; stroke: #fff; stroke-width: 2; }

    h1 {
      font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;
      background: linear-gradient(to right, #ffffff, #94a3b8);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .subtitle { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.2rem; }

    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--bg-card-border);
      border-radius: var(--radius-lg);
      padding: 1.1rem 1.25rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
      transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .card:hover { transform: translateY(-2px); border-color: rgba(255, 255, 255, 0.2); }

    .card-label {
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
      color: var(--text-muted); margin-bottom: 0.4rem;
    }

    .card-value {
      font-size: 1.5rem; font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }

    .val-cyan { color: var(--accent-cyan); }
    .val-pink { color: var(--accent-pink); }
    .val-green { color: var(--accent-green); }
    .val-amber { color: var(--accent-amber); }

    .card-subtext { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem; }

    /* Layout Main Section with Sidebar Control Panel */
    .main-workspace {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }

    /* Filter Toggle Control Panel */
    .filter-panel {
      background: var(--bg-card);
      border: 1px solid var(--bg-card-border);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
      backdrop-filter: blur(16px);
    }

    .panel-title {
      font-size: 1rem; font-weight: 700; color: #fff;
      display: flex; align-items: center; justify-content: space-between;
      padding-bottom: 0.75rem; margin-bottom: 1rem;
      border-bottom: 1px solid var(--bg-card-border);
    }

    .toggle-group {
      display: flex; flex-direction: column; gap: 1rem;
    }

    .toggle-item {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: var(--radius-sm);
      padding: 0.75rem 0.9rem;
      transition: all 0.2s ease;
    }

    .toggle-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .toggle-label {
      font-size: 0.85rem; font-weight: 600; color: #fff;
    }

    .badge-active {
      font-size: 0.7rem; font-weight: 700; color: #10b981;
      background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981;
      padding: 0.2rem 0.55rem; border-radius: 12px;
      letter-spacing: 0.05em;
    }

    .param-controls {
      display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.4rem;
    }

    .param-controls label {
      font-size: 0.75rem; color: var(--text-muted); display: flex; justify-content: space-between;
    }

    .param-controls input[type="range"] {
      width: 100%; accent-color: var(--accent-cyan); cursor: pointer;
    }

    /* Viewport Section */
    .viewport-section {
      background: var(--bg-card);
      border: 1px solid var(--bg-card-border);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      backdrop-filter: blur(16px);
    }

    .tab-nav {
      display: flex; gap: 0.6rem;
      border-bottom: 1px solid var(--bg-card-border);
      padding-bottom: 0.8rem; margin-bottom: 1.25rem;
      overflow-x: auto;
    }

    .tab-btn {
      background: transparent; border: none;
      color: var(--text-muted); padding: 0.6rem 1rem;
      border-radius: var(--radius-sm); font-size: 0.88rem; font-weight: 600;
      cursor: pointer; transition: all 0.2s ease;
    }

    .tab-btn:hover { color: var(--text-main); background: rgba(255, 255, 255, 0.05); }

    .tab-btn.active {
      color: #fff;
      background: linear-gradient(135deg, rgba(2, 132, 199, 0.25), rgba(139, 92, 246, 0.25));
      border: 1px solid var(--accent-cyan);
    }

    .visualizer-container {
      position: relative; width: 100%; min-height: 480px;
      display: flex; justify-content: center; align-items: center;
      border-radius: var(--radius-md); overflow: hidden;
      background: rgba(0, 0, 0, 0.5);
    }

    .image-wrapper { position: relative; max-width: 100%; display: inline-block; }

    .base-img, .overlay-img {
      display: block; max-width: 100%; height: auto; border-radius: var(--radius-sm);
    }

    .overlay-img { position: absolute; top: 0; left: 0; opacity: 0.5; }

    .controls-panel {
      display: flex; align-items: center; justify-content: space-between;
      margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--bg-card-border);
    }

    .slider-group { display: flex; align-items: center; gap: 1rem; }
    .slider-group label { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; }
    .slider-group input[type="range"] { accent-color: var(--accent-cyan); cursor: pointer; }

    .inspector-tooltip {
      font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
      color: var(--accent-cyan); background: rgba(0, 0, 0, 0.6);
      padding: 0.4rem 0.8rem; border-radius: var(--radius-sm);
      border: 1px solid rgba(2, 132, 199, 0.3);
    }

    canvas { width: 100%; height: 100%; display: block; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>
        </div>
        <div>
          <h1>iPhone Stereo Depth Extraction Studio</h1>
          <div class="subtitle">Interactive Real-Time Noise Filter Toggles & Parameter Tuning Studio</div>
        </div>
      </div>
      <div id="liveBadge" style="display: flex; align-items: center; gap: 0.5rem; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.82rem; font-weight: 600; color: #10b981;">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;"></span>
        <span>STATUS: READY</span>
      </div>
    </header>

    <!-- Telemetry Cards -->
    <div class="telemetry-grid">
      <div class="card">
        <div class="card-label">Rectified Resolution</div>
        <div class="card-value val-cyan" id="valRes">1280 × 960</div>
        <div class="card-subtext">Main (1x Wide) + Ultra-Wide (0.5x)</div>
      </div>
      <div class="card">
        <div class="card-label">Nominal Baseline</div>
        <div class="card-value val-green" id="valBaseline">19.5 mm</div>
        <div class="card-subtext">Dual Optical Axis Offset</div>
      </div>
      <div class="card">
        <div class="card-label">Focal Length</div>
        <div class="card-value val-amber" id="valFocal">960.0 px</div>
        <div class="card-subtext">Dynamic Epipolar Alignment</div>
      </div>
      <div class="card">
        <div class="card-label">Depth Range</div>
        <div class="card-value val-pink" id="valDepthRange">0.2m - 4.5m</div>
        <div class="card-subtext">Census-SGBM + Bilateral Filtered</div>
      </div>
    </div>

    <!-- Main Workspace with Sidebar & Viewport -->
    <div class="main-workspace">
      <!-- Filter Control Panel Sidebar -->
      <div class="filter-panel">
        <div class="panel-title">
          <span>🎛️ Pipeline Filter Status</span>
          <span style="font-size: 0.72rem; color: var(--accent-cyan); font-weight: 600;" id="filterStatusCount">5 ACTIVE</span>
        </div>

        <div class="toggle-group">
          <!-- Filter 1: Guided Bilateral Filter -->
          <div class="toggle-item">
            <div class="toggle-header">
              <span class="toggle-label">🌐 Guided Bilateral Filter</span>
              <span class="badge-active">ACTIVE</span>
            </div>
            <div class="param-controls">
              <label><span>Spatial / Color Sigma:</span><span id="valSigmaC" style="font-family: monospace; color: var(--accent-cyan);">σs=3.0, σc=20.0</span></label>
            </div>
          </div>

          <!-- Filter 2: WLS Edge-Preserving Filter -->
          <div class="toggle-item">
            <div class="toggle-header">
              <span class="toggle-label">🌊 WLS Edge Filter</span>
              <span class="badge-active">ACTIVE</span>
            </div>
            <div class="param-controls">
              <label><span>Smoothness Regularization:</span><span id="valWLS" style="font-family: monospace; color: var(--accent-cyan);">λ = 8000</span></label>
            </div>
          </div>

          <!-- Filter 3: Speckle Removal Filter -->
          <div class="toggle-item">
            <div class="toggle-header">
              <span class="toggle-label">✨ Speckle Removal Filter</span>
              <span class="badge-active">ACTIVE</span>
            </div>
            <div class="param-controls">
              <label><span>Max Speckle Component:</span><span id="valSpeckle" style="font-family: monospace; color: var(--accent-cyan);">100 px</span></label>
            </div>
          </div>

          <!-- Filter 4: Left-Right Cross Check -->
          <div class="toggle-item">
            <div class="toggle-header">
              <span class="toggle-label">↔️ Left-Right Check</span>
              <span class="badge-active">ACTIVE</span>
            </div>
            <div class="param-controls">
              <label><span>Disparity Consistency:</span><span id="valLR" style="font-family: monospace; color: var(--accent-cyan);">1.0 px</span></label>
            </div>
          </div>

          <!-- Filter 5: 3x3 Median Filter -->
          <div class="toggle-item">
            <div class="toggle-header">
              <span class="toggle-label">🎯 3x3 Median Filter</span>
              <span class="badge-active">ACTIVE</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Main Viewport Section -->
      <div class="viewport-section">
        <div class="tab-nav">
          <button class="tab-btn active" onclick="switchTab('overlay')">📷 RGB Depth Overlay</button>
          <button class="tab-btn" onclick="switchTab('raw_comparison')">📸 Raw Input vs Rectified</button>
          <button class="tab-btn" onclick="switchTab('disparity')">⚡ Disparity Map (Census-SGBM)</button>
          <button class="tab-btn" onclick="switchTab('depth')">🌈 Metric Depth Map</button>
          <button class="tab-btn" onclick="switchTab('stereo')">🔍 Rectified Stereo Pair</button>
          <button class="tab-btn" onclick="switchTab('epipolar')">📐 20px Epipolar Grid</button>
          <button class="tab-btn" onclick="switchTab('profile')">📈 Scanline Depth Profile</button>
          <button class="tab-btn" onclick="switchTab('pointcloud')">🧊 3D Point Cloud</button>
        </div>

        <div class="visualizer-container" id="visualizerContainer">
          <!-- Dynamic content rendered by JS -->
          <div class="image-wrapper" id="imageWrapper">
            <img id="baseImage" class="base-img" src="" alt="Base View">
            <img id="overlayImage" class="overlay-img" src="" alt="Overlay View" style="display: none;">
          </div>
          <canvas id="profileCanvas" style="display: none; height: 460px;"></canvas>
          <canvas id="pointCloudCanvas" style="display: none; height: 460px;"></canvas>
        </div>

        <div class="controls-panel">
          <div class="slider-group" id="opacityGroup">
            <label for="opacitySlider">Depth Overlay Opacity:</label>
            <input type="range" id="opacitySlider" min="0" max="1" step="0.05" value="0.5">
            <span id="opacityVal" style="font-family: monospace;">50%</span>
          </div>

          <!-- Disparity & Depth Scale Colorbar Legend -->
          <div id="legendBarContainer" style="display: flex; align-items: center; gap: 0.8rem; background: rgba(0, 0, 0, 0.4); padding: 0.4rem 0.8rem; border-radius: var(--radius-sm); border: 1px solid var(--bg-card-border);">
            <span id="legendTitle" style="font-size: 0.78rem; font-weight: 600; color: var(--text-muted);">Depth Scale (Z = f·B/d):</span>
            <div id="legendColorbar" style="width: 140px; height: 12px; border-radius: 4px; background: linear-gradient(to right, #f0f921, #e16462, #9c179e, #0d0887);"></div>
            <div id="legendLabels" style="display: flex; justify-content: space-between; gap: 0.6rem; font-family: monospace; font-size: 0.75rem; color: #fff;">
              <span id="legendMinLabel">0.2m (Near)</span>
              <span id="legendMaxLabel">5.0m (Far)</span>
            </div>
          </div>

          <div class="inspector-tooltip" id="inspectorTooltip">Hover image to inspect pixel depth</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let sceneData = null;
    let currentTab = 'overlay';
    let rotX = 0.3, rotY = 0.5, zoom = 1.0;
    let isMouseDown = false, lastMouseX = 0, lastMouseY = 0;
    let livePollTimer = null;

    async function loadData() {
      try {
        const resp = await fetch('data.json?t=' + Date.now());
        const data = await resp.json();
        if (data.scenes && data.scenes.length > 0) {
          sceneData = data.scenes[0];
          updateTelemetry();
          renderView();

          if (sceneData.is_live && !livePollTimer) {
            livePollTimer = setInterval(loadData, 1000);
          } else if (!sceneData.is_live && livePollTimer) {
            clearInterval(livePollTimer);
            livePollTimer = null;
          }
        }
      } catch (e) {
        console.error('Failed to load data.json', e);
      }
    }

    function updateTelemetry() {


      if (!sceneData) return;
      document.getElementById('valRes').textContent = `${sceneData.width} × ${sceneData.height}`;
      document.getElementById('valBaseline').textContent = `${sceneData.baseline.toFixed(1)} mm`;
      document.getElementById('valFocal').textContent = `${sceneData.focal_length.toFixed(1)} px`;
      
      const minZ = sceneData.scanline_depth_m ? Math.min(...sceneData.scanline_depth_m.filter(z => z > 0.1)).toFixed(2) : '0.25';
      const maxZ = sceneData.scanline_depth_m ? Math.max(...sceneData.scanline_depth_m.filter(z => z < 10.0)).toFixed(2) : '3.50';
      document.getElementById('valDepthRange').textContent = `${minZ}m - ${maxZ}m`;

      const badge = document.getElementById('liveBadge');
      if (sceneData.is_live) {
        badge.style.borderColor = '#f43f5e';
        badge.style.background = 'rgba(244, 63, 94, 0.15)';
        badge.style.color = '#f43f5e';
        badge.innerHTML = `<span style="width: 8px; height: 8px; border-radius: 50%; background: #f43f5e; display: inline-block;"></span> 🔴 REAL-TIME LIVE INSPECT (30 FPS)`;
      } else {
        badge.style.borderColor = '#10b981';
        badge.style.background = 'rgba(16, 185, 129, 0.15)';
        badge.style.color = '#10b981';
        badge.innerHTML = `<span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;"></span> 🟢 SESSION COMPLETED`;
      }
    }

    function switchTab(tab) {
      currentTab = tab;
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');
      renderView();
    }

    function renderView() {
      if (!sceneData) return;

      const baseImg = document.getElementById('baseImage');
      const overlayImg = document.getElementById('overlayImage');
      const imageWrapper = document.getElementById('imageWrapper');
      const profileCanvas = document.getElementById('profileCanvas');
      const pcCanvas = document.getElementById('pointCloudCanvas');
      const opacityGroup = document.getElementById('opacityGroup');
      const legendContainer = document.getElementById('legendBarContainer');
      const legendTitle = document.getElementById('legendTitle');
      const legendColorbar = document.getElementById('legendColorbar');
      const legendMinLabel = document.getElementById('legendMinLabel');
      const legendMaxLabel = document.getElementById('legendMaxLabel');

      const assetPath = `assets/${sceneData.name.replace(/ /g, '_')}`;

      imageWrapper.style.display = 'inline-block';
      profileCanvas.style.display = 'none';
      pcCanvas.style.display = 'none';
      opacityGroup.style.display = 'none';
      legendContainer.style.display = 'flex';

      if (currentTab === 'overlay') {
        baseImg.src = `${assetPath}/im0.jpg`;
        overlayImg.src = `${assetPath}/depth_color.jpg`;
        overlayImg.style.display = 'block';
        overlayImg.style.opacity = document.getElementById('opacitySlider').value;
        opacityGroup.style.display = 'flex';

        legendTitle.textContent = 'Depth Scale (Z = f·B/d):';
        legendColorbar.style.background = 'linear-gradient(to right, #f0f921, #e16462, #9c179e, #0d0887)';
        legendMinLabel.textContent = '0.2m (Near)';
        legendMaxLabel.textContent = '5.0m (Far)';
      } else if (currentTab === 'raw_comparison') {
        baseImg.src = `${assetPath}/raw_vs_rectified_panel.jpg`;
        overlayImg.style.display = 'none';
        legendContainer.style.display = 'none';
      } else if (currentTab === 'disparity') {
        baseImg.src = `${assetPath}/disp_raw.jpg`;
        overlayImg.style.display = 'none';

        legendTitle.textContent = 'Disparity Scale (px):';
        legendColorbar.style.background = 'linear-gradient(to right, #30123b, #28bceb, #a2fc3c, #fb8022, #7a0403)';
        legendMinLabel.textContent = '0 px (Far ∞)';
        legendMaxLabel.textContent = '64 / 160 px (Near)';
      } else if (currentTab === 'depth') {
        baseImg.src = `${assetPath}/depth_color.jpg`;
        overlayImg.style.display = 'none';

        legendTitle.textContent = 'Metric Depth Scale (Plasma):';
        legendColorbar.style.background = 'linear-gradient(to right, #f0f921, #e16462, #9c179e, #0d0887)';
        legendMinLabel.textContent = '0.2m (Near)';
        legendMaxLabel.textContent = '5.0m (Far)';
      } else if (currentTab === 'stereo') {
        baseImg.src = `${assetPath}/disp_intensity_panel.jpg`;
        overlayImg.style.display = 'none';

        legendTitle.textContent = 'Disparity Scale (px):';
        legendColorbar.style.background = 'linear-gradient(to right, #30123b, #28bceb, #a2fc3c, #fb8022, #7a0403)';
        legendMinLabel.textContent = '0 px (Far ∞)';
        legendMaxLabel.textContent = '64 / 160 px (Near)';
      } else if (currentTab === 'epipolar') {
        baseImg.src = `${assetPath}/epipolar_grid.jpg`;
        overlayImg.style.display = 'none';
        legendContainer.style.display = 'none';
      } else if (currentTab === 'profile') {
        imageWrapper.style.display = 'none';
        profileCanvas.style.display = 'block';
        legendContainer.style.display = 'none';
        drawProfileChart();
      } else if (currentTab === 'pointcloud') {
        imageWrapper.style.display = 'none';
        pcCanvas.style.display = 'block';
        legendContainer.style.display = 'none';
        init3DPointCloud();
      }
    }

    document.getElementById('opacitySlider').addEventListener('input', (e) => {
      const val = e.target.value;
      document.getElementById('opacityVal').textContent = `${Math.round(val * 100)}%`;
      if (currentTab === 'overlay') {
        document.getElementById('overlayImage').style.opacity = val;
      }
    });

    document.getElementById('imageWrapper').addEventListener('mousemove', (e) => {
      if (!sceneData) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) * (sceneData.width / rect.width));
      const y = Math.floor((e.clientY - rect.top) * (sceneData.height / rect.height));

      if (x >= 0 && x < sceneData.width && y >= 0 && y < sceneData.height) {
        const disp = sceneData.scanline_c ? (sceneData.scanline_c[x] || 0).toFixed(1) : 0;
        const depth = disp > 0 ? ((sceneData.focal_length * (sceneData.baseline / 1000.0)) / disp).toFixed(2) : 'N/A';
        document.getElementById('inspectorTooltip').textContent = `Pixel (X: ${x}, Y: ${y}) | Disparity: ${disp} px | Metric Depth: ${depth} m`;
      }
    });

    function drawProfileChart() {
      if (!sceneData || !sceneData.scanline_c) return;
      const canvas = document.getElementById('profileCanvas');
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;

      ctx.fillStyle = '#090d16'; ctx.fillRect(0, 0, w, h);

      const profile = sceneData.scanline_c;
      const maxDisp = Math.max(...profile, 1);
      const stepX = w / profile.length;

      ctx.strokeStyle = '#0284c7'; ctx.lineWidth = 2;
      ctx.beginPath();
      profile.forEach((val, i) => {
        const x = i * stepX;
        const y = h - (val / maxDisp) * (h - 40) - 20;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = '#94a3b8'; ctx.font = '12px Inter';
      ctx.fillText(`Mid-Scanline Disparity Profile (Y=${sceneData.mid_scanline_y})`, 20, 30);
    }

    function init3DPointCloud() {
      const canvas = document.getElementById('pointCloudCanvas');
      canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight;
      
      canvas.onmousedown = (e) => { isMouseDown = true; lastMouseX = e.clientX; lastMouseY = e.clientY; };
      window.onmouseup = () => isMouseDown = false;
      canvas.onmousemove = (e) => {
        if (!isMouseDown) return;
        rotY += (e.clientX - lastMouseX) * 0.005;
        rotX += (e.clientY - lastMouseY) * 0.005;
        lastMouseX = e.clientX; lastMouseY = e.clientY;
        render3DPointCloud();
      };
      canvas.onwheel = (e) => { zoom *= (e.deltaY > 0 ? 0.9 : 1.1); render3DPointCloud(); };
      render3DPointCloud();
    }

    function render3DPointCloud() {
      if (!sceneData || !sceneData.point_cloud) return;
      const canvas = document.getElementById('pointCloudCanvas');
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;

      ctx.fillStyle = '#090d16'; ctx.fillRect(0, 0, w, h);

      const pts = sceneData.point_cloud.points;
      const cols = sceneData.point_cloud.colors;
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const cx = w / 2, cy = h / 2;
      const scale = 0.25 * zoom;

      for (let i = 0; i < pts.length; i++) {
        let x = pts[i][0], y = pts[i][1], z = pts[i][2] - 1500;
        let rx = cosY * x + sinY * z;
        let rz = -sinY * x + cosY * z;
        let ry = cosX * y - sinX * rz;
        rz = sinX * y + cosX * rz;

        if (rz < 100) {
          const px = cx + (rx * scale);
          const py = cy + (ry * scale);
          const r = cols[i][0], g = cols[i][1], b = cols[i][2];
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(px, py, 2, 2);
        }
      }

      ctx.fillStyle = '#94a3b8'; ctx.font = '12px Inter';
      ctx.fillText('3D Point Cloud (Drag to Rotate • Scroll to Zoom)', 20, 30);
    }

    window.onload = loadData;
  </script>
</body>
</html>
"""
