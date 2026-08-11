# Depth Information Extraction Context

System for extracting 3D spatial depth maps from stereo image pairs using epipolar sliding window block matching and rendering interactive visual digests.

## Language

**Stereo Pair**:
A set of synchronized left (`im0.png`) and right (`im1.png`) image views captured by parallel optical sensors.
_Avoid_: Dual images, stereo frames

**Stereo Calibration**:
The estimated intrinsic parameters of both cameras and the rigid transform between them, expressed with explicit image resolutions and physical units.
_Avoid_: Calibration file, camera setup

**Calibration Limitation**:
An evidence-backed condition showing that the available calibration captures cannot satisfy the trusted Stereo Calibration contract, distinct from a downstream processing failure.
_Avoid_: Calibration pass, rectification failure, acceptable calibration

**Calibration Result**:
The evidence-backed outcome of assessing a declared calibration target and capture set. A Calibration Result is trusted, input-limited, or rejected; it contains an active Stereo Calibration only when trusted.
_Avoid_: Calibration file, best candidate, solver output

**Diagnostic Calibration Candidate**:
An explicitly untrusted set of camera parameters retained inside an input-limited Calibration Result solely to support limitation-aware diagnostic processing. It is never an active Stereo Calibration and cannot establish Trusted Depth Eligibility.
_Avoid_: Active calibration, fallback calibration, best calibration

**Main-to-Ultra-Wide Transform**:
The directed rigid transform that maps a 3D point in the Main camera coordinate frame into the Ultra-Wide camera coordinate frame, with translation expressed in meters.
_Avoid_: Extrinsics, camera transform, Ultra-Wide-to-Main transform

**Stereo Rectification**:
The geometric transformation of a Stereo Pair into views whose corresponding scene points lie on the same horizontal scanlines.
_Avoid_: Image alignment, image warping, retification

**Rectified Stereo Pair**:
A Stereo Pair in one shared output coordinate space whose physical left view is Ultra-Wide and physical right view is Main, accompanied by rectified projection geometry and geometric validity.
_Avoid_: Rectified images, aligned pair, Main-first pair

**Rectification Validity Mask**:
A binary matrix identifying output pixels whose remap samples valid source-image coordinates, independent of image brightness and downstream disparity validity.
_Avoid_: Nonblack mask, Disparity Validity Mask, overlap fill

**Epipolar Matching**:
1D horizontal sliding-window search along rectified scanlines to locate corresponding features between left and right views.
_Avoid_: 2D patch search, template scanning

**Disparity Map**:
A matrix of pixel offset values representing the horizontal shift ($x_L - x_R$) between matched points in a stereo pair.
_Avoid_: Shift matrix, offset map

**Disparity Confidence Map**:
A normalized matrix from 0 to 1 expressing the strength of correspondence evidence for each value in a Disparity Map, distinct from the binary validity decision.
_Avoid_: Confidence mask, validity mask, match score

**Disparity Validity Mask**:
A binary matrix identifying which Disparity Map values satisfy hard geometric and measurement requirements, independent of their graded confidence.
_Avoid_: Confidence mask, fill mask, positive-disparity mask

**Displayed Pixel Set**:
The page-local subset of hard-valid disparity/depth pixels whose normalized confidence meets the human-selected inspection threshold. Changing it affects only Digest Dashboard presentation and never changes stored computation or Trusted Depth Eligibility.
_Avoid_: Filtered disparity, new validity mask, recomputed depth

**Trusted Depth Eligibility**:
The evidence-backed status that Stereo Calibration, Stereo Rectification, Epipolar Matching, confidence, and validity have all met their acceptance contracts, allowing derived metric depth to be presented as trusted. Diagnostic availability alone does not establish Trusted Depth Eligibility.
_Avoid_: Has depth values, plausible depth, high fill rate, confidence threshold

**Stereo Processing Result**:
The named outcome of applying the shared calibration, Stereo Rectification, Epipolar Matching, confidence, validity, and metric-depth core to one Stereo Pair. A processed result is trusted or diagnostic and retains stage evidence plus inherited limitations; a rejected result contains diagnostics but no usable numerical artifacts.
_Avoid_: Pipeline output, depth result, success response

**Disparity Intensity Map**:
A grayscale image representation where pixel luminance/brightness directly maps to the magnitude of horizontal disparity ($I \propto d$).
_Avoid_: Grayscale depth, raw brightness image

**Epipolar Patch Visualizer**:
A visual diagram illustrating corresponding feature blocks between left and right stereo views connected by epipolar search lines.
_Avoid_: Box connector, search line diagram

**Depth Map**:
A matrix of physical metric distance values derived from focal length, baseline, and disparity ($Z = \frac{f \cdot B}{d + \text{doffs}}$).
_Avoid_: Range map, distance matrix

**Depth Information Extraction**:
The pipeline segment that estimates a Disparity Map from a rectified Stereo Pair and converts valid disparities into a metric Depth Map.
_Avoid_: Calibration, rectification, dashboard rendering

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
