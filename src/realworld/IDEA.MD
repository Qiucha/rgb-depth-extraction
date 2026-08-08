## 1. iPhone Camera Technical Specs (Direction 1)

For depth calculation on a single phone, you will utilize the **Main (Wide)** and **Ultra-Wide** camera modules.

* **Baseline Distance ($B$):** Approximately **1.5 cm – 2.0 cm** between the center of the Main lens and Ultra-Wide lens.
* **Optical Heterogeneity:**
* **Main Camera:** ~26mm equivalent focal length ($f_1 \approx 5.7\text{mm}$ physical), $f/1.6$ aperture, $\sim 80^\circ$ FOV.
* **Ultra-Wide Camera:** ~13mm equivalent focal length ($f_2 \approx 2.2\text{mm}$ physical), $f/2.2$ aperture, $\sim 120^\circ$ FOV.


* **Disparity Impact:** Because $f_1 \neq f_2$ and $\text{FOV}_1 \neq \text{FOV}_2$, the captured images cannot be directly fed into standard block matching. You must downsample/crop the Ultra-Wide image to match the Main camera's FOV and scale its focal length in software before calculating disparity.

---

## 2. iPhone Depth Extraction Pipeline

To achieve accurate depth, bypass static manual calibration (like checkerboards) when possible and leverage iOS's native dynamic calibration framework.

```
  [Capture Dual Video Buffers] (AVCaptureMultiCamSession)
               │
               ├──► Extract Dynamic Intrinsics (K1, K2) via AVCameraCalibrationData
               └──► Extract Extrinsics (R, T) via AVCaptureDevice.extrinsicMatrix
               │
  [FOV Alignment & Resampling] (Crop Ultra-Wide to Main FOV)
               │
  [Rectification & Normalization] (Align epipolar lines)
               │
  [Stereo Matching Algorithm] (SGBM or Deep Disparity Model e.g., CreStereo / AnyStereo)
               │
  [Depth Map Metric Calculation] Z = (f * B) / Disparity

```

### Key Engineering Steps:

1. **Synchronized Frame Capture:** Use `AVCaptureMultiCamSession` in iOS to stream frames simultaneously from `builtInWideAngleCamera` and `builtInUltraWideCamera`.
2. **Dynamic Calibration Retrieval:** Optical Image Stabilization (OIS) and autofocus shift lens elements dynamically. Do not hardcode static $K$ matrices. Instead, enable `isCameraIntrinsicMatrixDeliveryEnabled` on `AVCaptureConnection` to read $K_1, K_2$ per-frame via `AVCameraCalibrationData`. Obtain the hardware baseline extrinsic $T$ using `AVCaptureDevice.extrinsicMatrix(from:to:)`.
3. **Rectification & Matching:**
* Downsample the Ultra-Wide image to map to the Main camera's focal length.
* Apply OpenCV `cv2.stereoRectify` using the dynamic $K_1, K_2, R, T$ to produce coplanar, horizontally aligned epipolar lines.
* Execute disparity matching using **Semi-Global Block Matching (SGBM)** or a neural stereo network (such as *StereoStereo* or *RAFT-Stereo*) adapted for heterogeneous stereo pairs.



---

## 3. Comparison Baseline: Intel RealSense

Using an Intel RealSense camera (e.g., D435 / D455) serves as your **Ground Truth Benchmark**.

### Structural Differences

| Metric / Feature | iPhone Dual RGB (Dir 1) | Intel RealSense (e.g., D435/D455) |
| --- | --- | --- |
| **Technology** | Passive Dual RGB | Active IR Stereo + IR Projector |
| **Baseline ($B$)** | $\sim 1.5 - 2.0\text{ cm}$ | $5\text{ cm}$ (D435) / $9.5\text{ cm}$ (D455) |
| **Sensors** | Heterogeneous RGB (Different $f$, FOV) | Homogeneous Identical IR Lenses |
| **Textureless Surfaces** | Fails on plain walls (needs RGB texture) | Works via IR dot projection pattern |
| **Optimal Range** | **$0.2\text{m} - 1.2\text{m}$** | **$0.5\text{m} - 3.0\text{m}+$** |
| **Shutter Type** | Rolling Shutter (Main/UW) | Global Shutter (IR sensors) |

---

## 4. Evaluation Strategy: iPhone vs. RealSense

To measure the performance of your custom iPhone implementation against the RealSense benchmark, run the following tests:

### A. Point-Cloud Alignment & Distance Error (MAE / RMSE)

1. Mount the iPhone and RealSense side-by-side on a fixed rig facing a flat wall at known distances ($0.5\text{m}, 1.0\text{m}, 1.5\text{m}, 2.0\text{m}$).
2. Align both depth maps into a shared 3D point cloud coordinate frame using ICP (Iterative Closest Point) in Open3D.
3. Compute the **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)** of the depth $Z$ against the ground truth plane:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (Z_{\text{iPhone}, i} - Z_{\text{RealSense}, i})^2}$$



### B. Texture Dependency Test

* Evaluate depth output on **textured surfaces** (e.g., posters, brick walls) vs. **textureless surfaces** (e.g., white whiteboard, smooth office wall).
* *Expected Result:* The iPhone implementation will degrade significantly on textureless walls due to lack of optical features for passive block matching, whereas RealSense uses its IR projector to fill textureless gaps.

### C. Edge Preserving & Boundary Artifacts

* Place objects with sharp edges (e.g., chairs, boxes) in front of a background.
* Measure **flying pixels** and **depth bleed** at object boundaries. Heterogeneous optical resampling on the iPhone often introduces edge blur in disparity maps.
