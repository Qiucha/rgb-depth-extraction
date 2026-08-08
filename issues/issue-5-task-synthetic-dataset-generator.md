# Issue #5: Task - Synthetic Heterogeneous Dual-Cam & RealSense Benchmark Test Data Generator

**Labels**: `wayfinder:task`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: [Issue #3](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-3-prototype-realworld-pipeline-architecture.md)

## Question

Build a python script (`src/realworld/dataset_generator.py`) that generates synthetic test datasets conforming to the schema defined in Issue #2, producing paired Main RGB, Ultra-Wide RGB (with different FOV and focal length), dynamic $K_1, K_2, R, T$ matrices, and synthetic RealSense ground truth depth maps / point clouds for offline testing and unit verification.

## Resolution

1. Implemented `src/realworld/dataset_generator.py` generating synthetic heterogeneous stereo pairs (Main $f_1 \approx 1400\text{px}$, Ultra-Wide $f_2 \approx 600\text{px}$, baseline $B = 19.5\text{mm}$) with foreground box geometry at $Z=1.0\text{m}$ and background plane at $Z=2.0\text{m}$.
2. Produced sample sequence dataset in `data/realworld_sample/`.
3. Verified with end-to-end integration test in `tests/test_realworld_pipeline.py`.

