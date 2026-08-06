# Feature Specification: Disparity-Based Depth Information Extraction & Interactive Digest Dashboard

## Problem Statement

Robotics and autonomous perception engineers need a fast, deterministic, and verifiable method to extract 3D spatial depth maps from 2D stereo image pairs using epipolar sliding window block matching. Without quantitative benchmarking and interactive visual telemetry, it is difficult to analyze the trade-offs between matching cost metrics (NCC vs. ZNCC), post-processing filters (Median vs. WLS filtering), and physical metric depth conversion across complex geometric scenes. Furthermore, users require a clear visualization of the core heuristic: higher pixel disparity between left and right camera views corresponds directly to closer 3D depth, represented both as physical distance and normalized pixel intensity maps.

## Solution

A modular stereo depth extraction pipeline paired with an interactive HTML digest dashboard. The system ingests stereo pair datasets (such as the Middlebury Stereo Benchmark), parses camera calibration parameters (`calib.txt`), reads ground-truth PFM disparity maps (`disp0.pfm`), and executes two distinct processing pipelines:
1. **Pipeline A**: 1D Epipolar Sliding-Window Matcher with Normalized Cross-Correlation (NCC) cost metric, Left-Right Consistency Occlusion Masking, and Median Filtering.
2. **Pipeline C**: 1D Epipolar Sliding-Window Matcher with Zero-mean Normalized Cross-Correlation (ZNCC) cost metric and Weighted Least Squares (WLS) / Guided Edge-Preserving Filtering.

The extracted disparity maps are converted into metric depth maps ($Z = \frac{f \cdot B}{d + \text{doffs}}$) and normalized Disparity Intensity Maps ($I \propto d$). An interactive HTML Digest Dashboard (`digest/index.html`) presents scene selectors, live RGB-Depth overlays with opacity controls, residual error heatmaps ($|D_A - D_C|$), 1D scanline depth profile charts, 3D spatial point cloud WebGL viewers, and quantitative evaluation metrics (RMSE, MAE, Bad-Pixel ratios, and computational latency).

## User Stories

1. As a robotics perception engineer, I want to extract disparity maps from stereo image pairs using epipolar sliding window block matching, so that I can accurately estimate object distances without relying on active sensors.
2. As a computer vision developer, I want to convert pixel disparity into metric 3D depth using focal length and baseline calibration parameters, so that physical distances can be used for spatial navigation.
3. As a researcher, I want to compare Normalized Cross-Correlation (NCC) and Zero-mean Normalized Cross-Correlation (ZNCC) matching metrics, so that I can evaluate similarity performance across varying lighting conditions.
4. As a vision developer, I want to apply Left-Right consistency validation, so that occluded regions visible in only one camera view are masked out as invalid.
5. As a vision developer, I want to apply Weighted Least Squares (WLS) edge-preserving filtering to raw disparity estimates, so that depth boundaries align cleanly with object contours in the RGB image.
6. As a perception engineer, I want to view a Grayscale Disparity Intensity Map where pixel brightness directly maps to disparity magnitude, so that I can visually verify the heuristic that larger disparity corresponds to closer objects.
7. As an engineer, I want an Epipolar Patch Visualizer diagram showing corresponding feature windows and connecting scanline arrows between stereo pairs, so that I can inspect match alignments.
8. As a QA engineer, I want to evaluate estimated disparity maps against ground-truth PFM files, so that I can calculate exact RMSE, MAE, and Bad-Pixel error ratios ($>1\text{px}$).
9. As a developer, I want an interactive HTML digest dashboard with a scene selector dropdown, so that I can seamlessly benchmark multiple Middlebury dataset scenes in one place.
10. As a user, I want an interactive RGB-Depth Overlay with a live opacity slider in the dashboard, so that I can inspect depth boundary alignment against camera scene images.
11. As a user, I want a pixel-level inspector tooltip in the dashboard, so that hovering over any image point reveals its exact $(X, Y)$ coordinates, disparity in pixels, and metric depth in millimeters.
12. As an analyst, I want a Residual Difference Heatmap displaying $|D_A - D_C|$, so that I can localize error hotspots between matching pipelines.
13. As an engineer, I want an interactive 1D horizontal scanline depth profile chart, so that I can plot disparity variations across image width along any target row.
14. As a robotics engineer, I want an interactive rotatable and zoomable 3D point cloud WebGL viewer in the dashboard, so that I can examine 3D spatial reconstructions.
15. As a developer, I want access to an inline domain model glossary in the dashboard, so that ubiquitous terminology (e.g., Epipolar Matching, Disparity Map, WLS Filtering) is clear to all stakeholders.

## Implementation Decisions

- **Stereo Ingestion & Calibration Parsing**: Module for reading Middlebury stereo dataset folders, extracting camera focal length $f$, baseline $B$, disparity offset `doffs`, maximum disparity `ndisp` from `calib.txt`, and single-precision 32-bit floating point arrays from `disp0.pfm`.
- **Epipolar Block Matching Engine**: Vectorized 1D horizontal scanline search module computing cost volumes for candidate disparities $d \in [\text{min\_disp}, \text{max\_disp}]$ across NCC, SAD, SSD, and ZNCC similarity metrics.
- **Sub-Pixel Refinement & Occlusion Masking**: Parabolic sub-pixel interpolation around cost volume extrema combined with vectorized Left-Right cross-checking for invalidating occluded boundary pixels.
- **Edge-Preserving Filtering Seam**: Guided image filtering / WLS regularization module smoothing raw disparity maps while locking edge boundaries to guidance image RGB intensity gradients.
- **Metric Depth & Point Cloud Projection**: Inverse disparity depth conversion ($Z = \frac{f \cdot B}{d + \text{doffs}}$) and pinhole camera 3D point projection ($X = \frac{(x - cx) \cdot Z}{f}$, $Y = \frac{(y - cy) \cdot Z}{f}$).
- **Single-Page HTML Digest Application**: Self-contained client-side dashboard in `digest/index.html` loading dataset telemetry from `digest/data.json` and static image assets from `digest/assets/`. Includes Canvas-based 1D scanline chart rendering and custom WebGL/Canvas 3D point cloud rotatable orbit controls.

## Testing Decisions

- **Testing Seam**: High-level integration seam evaluating `compute_disparity()` and `disparity_to_depth()` output matrices directly against ground-truth PFM float32 arrays parsed from benchmark scenes.
- **Good Test Criteria**: Tests must verify external mathematical contracts (RMSE, MAE, depth non-negativity, array bounds, sub-pixel continuity) rather than internal loop iterations.
- **Modules Tested**: `sliding_window.py`, `wls_filter.py`, `depth_calculator.py`, `evaluator.py`, `pfm_io.py`, `calib.py`.
- **Prior Art**: Middlebury Stereo Benchmark evaluation conventions (bad pixel threshold $> 1.0\text{px}$, PFM `inf` masking).

## Out of Scope

- Real-time video stream processing from live USB stereo camera hardware (e.g., stereo webcams).
- Deep learning neural network disparity estimation (e.g., PSMNet, StereoNet).
- Unrectified vertical alignment rectification (system assumes pre-rectified epipolar scanlines).

## Further Notes

- The primary testing seam is established at the `DepthEvaluator` level, comparing raw NumPy float outputs from `SlidingWindowMatcher` against `read_pfm()` ground-truth arrays.
- Performance testing confirms sub-500ms processing latency per stereo pair scene on downsampled benchmark resolutions.
