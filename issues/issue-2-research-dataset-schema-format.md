# Issue #2: Research - iPhone Dual-Cam & RealSense Benchmark Dataset Schema

**Labels**: `wayfinder:research`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Dataset Schema Researcher (`a039aab5-6688-4b54-8485-61455cb6b882`)  
**Blocks**: [Issue #3](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-3-prototype-realworld-pipeline-architecture.md), [Issue #4](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-4-grilling-icp-alignment-eval-strategy.md)  
**Blocked by**: None (Frontier)

## Question

What input data format & JSON/HDF5 directory schema should be defined for recording and loading real-world iPhone dual-camera captures (Main RGB, Ultra-Wide RGB, dynamic per-frame $K_1, K_2, R, T$) alongside Intel RealSense ground-truth reference depth maps / 3D point clouds (`.ply` / `.pcd` / numpy arrays)?

### Details to investigate
1. Schema representation for per-frame dynamic camera calibration data from iOS `AVCameraCalibrationData` (`intrinsicMatrix`, `extrinsicMatrix`, lens distortion lookup tables if any).
2. Format for Intel RealSense D435/D455 reference depth maps & exported 3D point clouds.
3. Compatibility with existing `data/` folder conventions in `src/stereo_depth/` (e.g. Middlebury `calib.txt`, `im0.png`, `im1.png`, `disp0.pfm`).
