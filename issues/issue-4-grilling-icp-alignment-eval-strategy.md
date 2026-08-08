# Issue #4: Grilling - Open3D ICP Point Cloud Alignment & Evaluation Strategy

**Labels**: `wayfinder:grilling`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: [Issue #1](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-1-research-hetero-fov-alignment.md), [Issue #2](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-2-research-dataset-schema-format.md)

## Question

What exact alignment protocol, initial rigid transformation estimation (e.g. initial $R, T$ alignment or global registration), and metric definitions (MAE, RMSE, Texture Dependency Error Ratio, Flying Pixel Boundary Bleed) will be used when comparing iPhone passive stereo point clouds against RealSense active IR ground truth in Open3D?

## Resolution

Decisions locked via grilling with user:
1. **ICP Protocol**: Point-to-Point ICP (`max_correspondence_distance=0.05m`) with SVD rigid transform fallback implemented in `RealSensePointcloudAligner`.
2. **Distance Binning**: Metric evaluation binned across 4 discrete target distance planes: 0.5m, 1.0m, 1.5m, and 2.0m.
3. **Metric Suite**: Multi-axis reporting: MAE (m), RMSE (m), Bad-Pixel Ratio (>5cm threshold), Texture Dependency Ratio (Sobel gradient error ratio $\text{MAE}_{\text{textureless}} / \text{MAE}_{\text{textured}}$), and Flying Pixel Boundary Ratio (Canny edge depth gradient bleed).

