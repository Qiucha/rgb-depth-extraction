# Issue #3: Prototype - RealWorld Pipeline Architecture Extending stereo_depth

**Labels**: `wayfinder:prototype`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: [Issue #5](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-5-task-synthetic-dataset-generator.md)  
**Blocked by**: [Issue #1](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-1-research-hetero-fov-alignment.md), [Issue #2](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-2-research-dataset-schema-format.md)

## Question

How should the Python codebase under `src/realworld/` be structured to cleanly reuse existing `src/stereo_depth/` modules (`SlidingWindowMatcher`, `WLSFilter`, `DepthCalculator`, `DepthEvaluator`, `DigestGenerator`) while adding heterogeneous FOV alignment, dynamic epipolar rectification, Open3D ICP alignment, and realworld benchmark evaluation?

## Resolution

Created prototype architecture in `src/realworld/`:
1. `hetero_rectifier.py`: `HeterogeneousStereoRectifier` performing single-pass `cv2.stereoRectify` remapping and scale-invariant depth conversion.
2. `dataset_loader.py`: `RealWorldDatasetLoader` parsing `dataset_manifest.json` and synthesizing Middlebury `calib.txt` adapter.
3. `realsense_icp.py`: `RealSensePointcloudAligner` projecting depth maps to 3D point clouds and performing Open3D ICP registration.
4. `evaluator.py`: `RealWorldEvaluator` measuring MAE, RMSE, bad-pixel ratio, texture dependency ratio, and flying pixels.
5. `pipeline.py`: `run_realworld_pipeline()` master orchestrator.
6. Verified with unit tests in `tests/test_realworld_pipeline.py`.

