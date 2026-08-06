# Depth Information Extraction Context

System for extracting 3D spatial depth maps from stereo image pairs using epipolar sliding window block matching and rendering interactive visual digests.

## Language

**Stereo Pair**:
A set of synchronized left (`im0.png`) and right (`im1.png`) image views captured by parallel optical sensors.
_Avoid_: Dual images, stereo frames

**Epipolar Matching**:
1D horizontal sliding-window search along rectified scanlines to locate corresponding features between left and right views.
_Avoid_: 2D patch search, template scanning

**Disparity Map**:
A matrix of pixel offset values representing the horizontal shift ($x_L - x_R$) between matched points in a stereo pair.
_Avoid_: Shift matrix, offset map

**Disparity Intensity Map**:
A grayscale image representation where pixel luminance/brightness directly maps to the magnitude of horizontal disparity ($I \propto d$).
_Avoid_: Grayscale depth, raw brightness image

**Epipolar Patch Visualizer**:
A visual diagram illustrating corresponding feature blocks between left and right stereo views connected by epipolar search lines.
_Avoid_: Box connector, search line diagram

**Depth Map**:
A matrix of physical metric distance values derived from focal length, baseline, and disparity ($Z = \frac{f \cdot B}{d + \text{doffs}}$).
_Avoid_: Range map, distance matrix

**Cost Metric**:
Mathematical function (NCC, SAD, SSD, ZNCC) used to compute pixel block similarity across scanlines.
_Avoid_: Match score, distance function

**Occlusion Masking**:
Detection and invalidation of pixels visible in only one optical view using Left-Right cross-checking.
_Avoid_: Shadow masking, hole filling

**WLS Filtering**:
Weighted Least Squares edge-preserving smoothing applied to disparity estimates to align disparity boundaries with image gradients.
_Avoid_: Edge filter, smooth pass

**Middlebury Dataset**:
Standardized benchmark dataset containing stereo pairs, PFM ground-truth disparity maps, and `calib.txt` parameters.
_Avoid_: Test folder, raw image set

**PFM Format**:
Portable Float Map storing 32-bit single-precision floating point ground-truth disparity values and infinity markers.
_Avoid_: Float image, raw map file

**Digest Dashboard**:
An interactive single-page HTML application presenting stereo depth maps, residual heatmaps, performance benchmarks, and 3D point clouds.
_Avoid_: Web report, status page

**Residual Difference Heatmap**:
A color-coded 2D map displaying pixel-by-pixel absolute difference ($|D_A - D_C|$) between disparity outputs.
_Avoid_: Delta map, error map

**Depth Profile**:
A 1D spatial plot along a horizontal scanline displaying depth continuous variations across image width.
_Avoid_: Scanline graph, spatial slice

**Depth Overlay**:
A visual composite blending the original RGB camera image with a color-mapped depth map using adjustable opacity.
_Avoid_: Image tint, alpha mask
