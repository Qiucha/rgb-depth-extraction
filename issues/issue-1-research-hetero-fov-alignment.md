# Issue #1: Research - Heterogeneous FOV Alignment & Dynamic Rectification Math

**Labels**: `wayfinder:research`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: FOV Alignment Researcher (`da7d5d65-2d06-468c-a1bc-d916902b61c3`)  
**Blocks**: [Issue #3](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-3-prototype-realworld-pipeline-architecture.md), [Issue #4](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-4-grilling-icp-alignment-eval-strategy.md)  
**Blocked by**: None (Frontier)

## Question

How should the heterogeneous Ultra-Wide camera frame ($f_2 \approx 2.2\text{mm}, 120^\circ \text{FOV}$) be downsampled, cropped, and scaled to match the Main camera frame ($f_1 \approx 5.7\text{mm}, 80^\circ \text{FOV}$) prior to running OpenCV `cv2.stereoRectify` with dynamic intrinsic matrices $K_1, K_2$ and baseline extrinsics $R, T$?

### Details to investigate
1. Mathematical transformation for FOV alignment & scaling between Main ($K_1$) and Ultra-Wide ($K_2$).
2. How `cv2.stereoRectify` handles intrinsic scale changes when Ultra-Wide image is cropped/resampled.
3. Verification of disparity-to-depth formula $Z = \frac{f \cdot B}{d}$ when focal lengths $f_1 \neq f_2$ are normalized to rectified virtual focal length $f_{\text{rect}}$.
