# Feature Specification: 3D Room Stereo Camera Projection & Integrated Depth Extraction Pipeline

## Problem Statement

Virtual 3D environments and robotic simulation environments require synchronized stereo camera rendering to evaluate depth extraction algorithms under synthetic control. Previously, stereoscopic room rendering with pinhole camera models, trajectory waypoints, and customizable baseline offsets existed isolated in a standalone experiment folder (`3DCUBE`), separated from the main stereo depth extraction pipeline in `src/stereo_depth`. Without integrating the virtual 3D room environment and camera rig into the core codebase, perception developers could not interactively visualize live depth map extraction on synthetic 3D scenes in real time during camera motion or baseline adjustments.

## Solution

Integrate the 3D Room Projection and Binocular Stereo Camera Rig into the primary codebase as `src/cube_projection`, and connect it directly with `src/stereo_depth`'s epipolar sliding-window block matcher (`SlidingWindowMatcher`) and physical metric depth calculator (`DepthCalculator`). The integrated viewer (`StereoRoomVisualizer`) displays a 2x2 grid in Matplotlib:
1. **Top-Left**: Left Camera View (`im0`)
2. **Top-Right**: Right Camera View (`im1`)
3. **Bottom-Left**: Overlapped Realtime Stereo Composite (Left: Cyan, Right: Rose translucent overlay for parallax shift visualization)
4. **Bottom-Right**: Live Extracted 3D Depth Map directly computed via epipolar matching and metric depth conversion ($Z = \frac{f \cdot B}{d + \text{doffs}}$) on each render frame.

## User Stories

1. As a robotics engineer, I want a synthetic 3D room environment with enclosed walls, pedestals, and floating geometric shapes, so that I can evaluate stereo perception under controlled 3D room geometry.
2. As a computer vision developer, I want a binocular Stereo Camera Rig with configurable baseline distance $B$ and focal length $f$, so that I can simulate parallel optical sensors with strictly horizontal shifts.
3. As a user, I want interactive WASD/QE and Arrow key navigation, so that I can freely fly the stereo camera rig through the 3D room environment.
4. As a user, I want trajectory keyframe waypoints (1-5) and a smooth trajectory tour animation, so that I can evaluate depth map extraction along continuous camera movement paths.
5. As a developer, I want an overlapped stereo composite view rendering Left view in Cyan and Right view in Rose, so that horizontal parallax shifts between stereo views are visually apparent.
6. As a perception researcher, I want the bottom-right quadrant of the window to display a live extracted 3D depth map, so that I can verify epipolar block matching and depth calculations in real time.
7. As a developer, I want interactive focal length and baseline distance sliders, so that I can dynamically adjust camera parameters and observe immediate changes in the extracted depth map.
8. As a vision developer, I want an export feature (pressing 'P') to save synchronized Left (`im0.png`), Right (`im1.png`), and calibration parameters (`calib.txt`), so that synthetic scenes can be ingested by offline depth benchmarking scripts.
9. As an engineer, I want back-face culling and near-plane polygon clipping enabled on the pinhole cameras, so that room surfaces behind the camera or facing away are correctly culled without rendering artifacts.
10. As a software developer, I want the `3DCUBE/` folder to remain completely unchanged and isolated, so that legacy standalone experiments are preserved without side effects.
11. As a test engineer, I want automated unit tests verifying the integrated stereo depth visualizer, so that depth map matrix generation and pipeline execution are continuously validated.

## Implementation Decisions

- **Integrated Module Package (`src/cube_projection`)**: Reusable Python package containing `PinholeCamera`, `StereoCameraRig`, `RoomEnvironment`, `Face3D`, `Cube3D`, `NavigationController`, `RoomVisualizer`, and `StereoRoomVisualizer`.
- **Pipeline Integration**: `StereoRoomVisualizer` imports `SlidingWindowMatcher` and `DepthCalculator` directly from `src.stereo_depth`. On each render update, projected 3D faces are rasterized into 2D BGR image arrays (`img_left` and `img_right`) and passed to `compute_disparity()` and `disparity_to_depth()`.
- **Bottom-Right Quadrant Depth Renderer**: The bottom-right subplot (`self.ax_depth`) renders the resulting float32 metric depth array using `imshow` with a `plasma` colormap and an interactive depth scale colorbar.
- **Strict Folder Isolation**: All integrated modules live inside `src/cube_projection/`, ensuring `3DCUBE/` remains untouched.

## Testing Decisions

- **Testing Seam**: High-level visualizer seam (`TestIntegratedStereoDepth` in `tests/test_integrated_stereo_depth.py`) invoking `StereoRoomVisualizer.update()` and checking that `visualizer.depth_im.get_array()` returns a valid 2D numpy float depth array matching screen dimensions ($H \times W$) with non-zero depth values.
- **Good Test Criteria**: Test verifies end-to-end integration without mocking internal camera projections or block matching loops.
- **Modules Tested**: `src/cube_projection/stereo_visualizer.py`, `src/cube_projection/camera.py`, `src/stereo_depth/sliding_window.py`, `src/stereo_depth/depth_calculator.py`.

## Out of Scope

- GPU-accelerated CUDA stereo matching (pipeline uses NumPy/OpenCV CPU box filtering).
- Active lighting or shadows in the 3D room environment (uses Lambertian directional shading).
- Real-time video input from physical dual webcam hardware.

## Further Notes

- The 1D epipolar sliding window matcher operates with `window_size=7` and `max_disparity=48`, yielding ~30 FPS interactive rendering speed.
