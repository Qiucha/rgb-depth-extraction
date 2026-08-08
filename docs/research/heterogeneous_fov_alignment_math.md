# Research Report: Issue #1 — Heterogeneous FOV Alignment & Dynamic Rectification Math

**Ticket**: [Issue #1: Research - Heterogeneous FOV Alignment & Dynamic Rectification Math](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-1-research-hetero-fov-alignment.md)  
**Status**: Research Completed & Verified  

---

## Executive Summary

This research report investigates the geometric, mathematical, and algorithmic pipeline required to align, downsample, crop, and rectify a heterogeneous passive stereo pair captured by an iPhone dual-camera system:
- **Main Camera (Wide)**: $f_1 \approx 5.7\text{mm}$, $\text{FOV}_1 \approx 80^\circ$, typical aperture $f/1.6$.
- **Ultra-Wide Camera**: $f_2 \approx 2.2\text{mm}$, $\text{FOV}_2 \approx 120^\circ$, typical aperture $f/2.2$.
- **Baseline ($B$)**: Approximately $1.5\text{cm} - 2.0\text{cm}$ ($0.015\text{m} - 0.020\text{m}$).

### Key Findings
1. **Unified Single-Pass Remapping is Optimal**: Explicit pre-cropping and resizing of the Ultra-Wide image in Python prior to rectification introduces double pixel interpolation blur and unnecessary memory overhead. Instead, feeding raw intrinsics ($K_1, K_2$), distortion coefficients ($D_1, D_2$), extrinsics ($R, T$), and target output size `newImageSize` directly into OpenCV's `cv2.stereoRectify` + `cv2.initUndistortRectifyMap` + `cv2.remap` achieves distortion correction, rotation alignment, FOV crop, and resolution scaling in **a single bilinear/bicubic sampling pass**.
2. **Intrinsic Update Rule for Pre-Cropping**: If pre-cropping is desired for memory optimization, cropping top-left $(u_0, v_0)$ and scaling by $(s_x, s_y)$ transforms $K_2 \to K_2' = \begin{bmatrix} s_x f_{x2} & 0 & s_x (c_{x2} - u_0) \\ 0 & s_y f_{y2} & s_y (c_{y2} - v_0) \\ 0 & 0 & 1 \end{bmatrix}$.
3. **Depth Scale Invariance**: The disparity-to-depth formula $Z = \frac{f_{\text{rect}} \cdot B}{d + \text{doffs}}$ is **strictly scale-invariant** under image downsampling. When images are scaled by factor $s$, both $f_{\text{rect}}' = s \cdot f_{\text{rect}}$ and $d' = s \cdot d$ scale by $s$, leaving depth $Z = \frac{(s \cdot f_{\text{rect}}) \cdot B}{s \cdot d} = \frac{f_{\text{rect}} \cdot B}{d}$ unchanged.
4. **Dynamic Intrinsics (iOS AVFoundation)**: Because Optical Image Stabilization (OIS) and Auto-Focus (AF) shift lens elements dynamically, static calibration matrices must not be hardcoded. Per-frame calibration ($K_1, K_2, R, T$) delivered via `AVCameraCalibrationData` must be used.

---

## 1. Hardware Specs & Geometry of Heterogeneous Pair

| Sensor / Metric | Main Camera (Wide) | Ultra-Wide Camera | Ratio / Difference |
| --- | --- | --- | --- |
| **Focal Length ($f$)** | $f_1 \approx 5.7\text{mm}$ | $f_2 \approx 2.2\text{mm}$ | $\gamma = f_1 / f_2 \approx 2.59\times$ |
| **Horizontal FOV ($\theta$)** | $\approx 80^\circ$ | $\approx 120^\circ$ | $+40^\circ$ wider FOV on Ultra-Wide |
| **Sensor Resolution** | $4032 \times 3024$ (12 MP) | $4032 \times 3024$ (12 MP) | Equal pixel grid count |
| **Angular Pixel Density** | $\approx 50.4 \text{ px/deg}$ | $\approx 33.6 \text{ px/deg}$ | Main has $1.5\times$ higher angular resolution |
| **Baseline ($B$)** | Center-to-center baseline $B \approx 1.5 - 2.0\text{ cm}$ along optical rig |

### FOV Overlap Geometry
Because $\text{FOV}_2 > \text{FOV}_1$, only the central region of the Ultra-Wide sensor plane overlaps with the Main camera frame.
The angular coverage ratio:
$$\frac{\tan(\text{FOV}_1 / 2)}{\tan(\text{FOV}_2 / 2)} = \frac{\tan(40^\circ)}{\tan(60^\circ)} \approx \frac{0.8391}{1.7321} \approx 0.4844$$
Thus, approximately **$48.4\%$ of the horizontal spatial range** (and $\approx 38.6\%$ of the focal length proportion $f_2 / f_1$) of the Ultra-Wide frame contains the overlapping field of view. The outer $51.6\%$ has no stereo correspondence and must be cropped during rectification.

---

## 2. Mathematical Transformations for FOV Alignment, Crop & Scale

### A. Camera Intrinsics Model
Let the pinhole camera matrices for Main ($K_1$) and Ultra-Wide ($K_2$) be:
$$K_1 = \begin{bmatrix} f_{x1} & 0 & c_{x1} \\ 0 & f_{y1} & c_{y1} \\ 0 & 0 & 1 \end{bmatrix}, \quad K_2 = \begin{bmatrix} f_{x2} & 0 & c_{x2} \\ 0 & f_{y2} & c_{y2} \\ 0 & 0 & 1 \end{bmatrix}$$

### B. Spatial Crop Bounding Box on Ultra-Wide Sensor
To isolate the region of interest (ROI) on the Ultra-Wide sensor that corresponds to the Main camera's FOV:
1. **Crop Box Dimensions**:
   $$W_{\text{crop}, 2} = W_1 \cdot \frac{f_{x2}}{f_{x1}}, \quad H_{\text{crop}, 2} = H_1 \cdot \frac{f_{y2}}{f_{y1}}$$
2. **Crop Top-Left Corner $(u_{0,2}, v_{0,2})$**:
   $$u_{0,2} = c_{x2} - \frac{W_{\text{crop}, 2}}{2} = c_{x2} - \frac{f_{x2}}{f_{x1}} \cdot \frac{W_1}{2}$$
   $$v_{0,2} = c_{y2} - \frac{H_{\text{crop}, 2}}{2} = c_{y2} - \frac{f_{y2}}{f_{y1}} \cdot \frac{H_1}{2}$$

### C. Scaling Factors to Target Working Resolution
When scaling the cropped region to a target working resolution $(W_{\text{target}}, H_{\text{target}})$:
$$s_x = \frac{W_{\text{target}}}{W_{\text{crop}, 2}} = \frac{W_{\text{target}}}{W_1} \cdot \frac{f_{x1}}{f_{x2}}$$
$$s_y = \frac{H_{\text{target}}}{H_{\text{crop}, 2}} = \frac{H_{\text{target}}}{H_1} \cdot \frac{f_{y1}}{f_{y2}}$$

### D. Transformation of Intrinsic Matrix $K_2 \to K_2'$
Combining shift by $(-u_{0,2}, -v_{0,2})$ and scaling by $(s_x, s_y)$ yields the matrix operator:
$$M_{\text{crop\_scale}} = \begin{bmatrix} s_x & 0 & -s_x u_{0,2} \\ 0 & s_y & -s_y v_{0,2} \\ 0 & 0 & 1 \end{bmatrix}$$
Applying this to raw pixel coordinates $[u_2, v_2, 1]^T$ produces transformed coordinates $[u_2', v_2', 1]^T$.
The updated intrinsic matrix $K_2'$ is:
$$K_2' = M_{\text{crop\_scale}} K_2 = \begin{bmatrix} s_x f_{x2} & 0 & s_x (c_{x2} - u_{0,2}) \\ 0 & s_y f_{y2} & s_y (c_{y2} - v_{0,2}) \\ 0 & 0 & 1 \end{bmatrix}$$

Notice that when $W_{\text{target}} = W_1$:
$$f_{x2}' = s_x f_{x2} = \left(\frac{f_{x1}}{f_{x2}}\right) f_{x2} = f_{x1}$$
The effective focal length of the cropped and rescaled Ultra-Wide image matches the Main camera's focal length $f_{x1}$ exactly!

---

## 3. OpenCV `cv2.stereoRectify` and Single-Pass Unified Remapping

### A. Algorithm Flow of `cv2.stereoRectify`
OpenCV `cv2.stereoRectify` takes raw parameters:
```python
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1=K1, distCoeffs1=D1,
    cameraMatrix2=K2, distCoeffs2=D2,
    imageSize=(W1, H1),
    R=R, T=T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0,
    newImageSize=(W_target, H_target)
)
```

#### Output Projection Matrices:
$$P_1 = \begin{bmatrix} f_{\text{rect}} & 0 & c_{x1,\text{rect}} & 0 \\ 0 & f_{\text{rect}} & c_{y,\text{rect}} & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}, \quad P_2 = \begin{bmatrix} f_{\text{rect}} & 0 & c_{x2,\text{rect}} & T_x \cdot f_{\text{rect}} \\ 0 & f_{\text{rect}} & c_{y,\text{rect}} & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

- Both virtual cameras share a **common focal length** $f_{\text{rect}}$ ($f_{x,\text{rect}} = f_{y,\text{rect}} = f_{\text{rect}}$).
- Both virtual cameras share a **common vertical principal point** $c_{y,\text{rect}}$, ensuring epipolar lines are aligned along horizontal rows ($y_1 = y_2$).
- Horizontal baseline translation in $P_2$: $P_2[0,3] = -f_{\text{rect}} \cdot B$ (where $B = \|T\|$).

### B. Single-Pass Coordinate Mapping (`cv2.initUndistortRectifyMap` + `cv2.remap`)
To avoid double resampling, OpenCV evaluates mapping maps for each camera $i \in \{1, 2\}$:
```python
map_x, map_y = cv2.initUndistortRectifyMap(
    cameraMatrix=Ki, distCoeffs=Di, R=Ri, P=Pi,
    size=(W_target, H_target), m1type=cv2.CV_32FC1
)
rectified_img = cv2.remap(raw_img, map_x, map_y, cv2.INTER_LINEAR)
```

**Key Result**: Passing raw $K_2$ directly to `initUndistortRectifyMap` samples the central region of the Ultra-Wide frame, crops the non-overlapping FOV, corrects barrel distortion, rotates the optical axis, and scales to $f_{\text{rect}}$ **in one single bilinear interpolation step**.

---

## 4. Verification of Disparity-to-Depth Formula ($Z = \frac{f_{\text{rect}} \cdot B}{d + \text{doffs}}$)

### A. Mathematical Derivation
In rectified space, camera 1 is at origin $(0,0,0)^T$ and camera 2 is shifted by baseline vector $(B, 0, 0)^T$.
For any 3D scene point $(X, Y, Z)^T$:
- Left image x-projection: $u_1 = f_{\text{rect}} \frac{X}{Z} + c_{x1,\text{rect}}$
- Right image x-projection: $u_2 = f_{\text{rect}} \frac{X - B}{Z} + c_{x2,\text{rect}}$
- Disparity $d = u_1 - u_2$:
  $$d = \left(f_{\text{rect}} \frac{X}{Z} + c_{x1,\text{rect}}\right) - \left(f_{\text{rect}} \frac{X - B}{Z} + c_{x2,\text{rect}}\right) = \frac{f_{\text{rect}} \cdot B}{Z} + (c_{x1,\text{rect}} - c_{x2,\text{rect}})$$

Let disparity offset $\text{doffs} = c_{x2,\text{rect}} - c_{x1,\text{rect}}$:
$$d + \text{doffs} = \frac{f_{\text{rect}} \cdot B}{Z} \implies Z = \frac{f_{\text{rect}} \cdot B}{d + \text{doffs}}$$

### B. Proof of Depth Scale-Invariance Under Downsampling
Suppose the rectified image dimensions are scaled by factor $s \in (0, 1]$:
- Scaled rectified focal length: $f_{\text{rect}}' = s \cdot f_{\text{rect}}$
- Scaled pixel disparity: $d' = s \cdot d$

Recomputing metric depth $Z'$:
$$Z' = \frac{f_{\text{rect}}' \cdot B}{d'} = \frac{(s \cdot f_{\text{rect}}) \cdot B}{s \cdot d} = \frac{f_{\text{rect}} \cdot B}{d} = Z$$
**Conclusion**: Metric depth $Z$ is mathematically **100% invariant** to image resolution downsampling.

---

## 5. Reference Python Implementation Architecture

```python
import cv2
import numpy as np

class HeterogeneousStereoRectifier:
    """
    Handles FOV alignment, dynamic epipolar rectification, and disparity-to-depth
    conversion for heterogeneous iPhone stereo pairs (Main + Ultra-Wide).
    """

    def __init__(self, target_size=(1280, 960)):
        self.target_size = target_size

    def scale_intrinsics(self, K: np.ndarray, orig_size: tuple, target_size: tuple) -> np.ndarray:
        sx = target_size[0] / orig_size[0]
        sy = target_size[1] / orig_size[1]
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx  # fx
        K_scaled[1, 1] *= sy  # fy
        K_scaled[0, 2] *= sx  # cx
        K_scaled[1, 2] *= sy  # cy
        return K_scaled

    def rectify_pair(self, img_main: np.ndarray, img_uw: np.ndarray,
                     K1: np.ndarray, D1: np.ndarray,
                     K2: np.ndarray, D2: np.ndarray,
                     R: np.ndarray, T: np.ndarray):
        
        size1 = (img_main.shape[1], img_main.shape[0])
        size2 = (img_uw.shape[1], img_uw.shape[0])

        # 1. Scale intrinsics to working resolution
        K1_scaled = self.scale_intrinsics(K1, size1, self.target_size)
        K2_scaled = self.scale_intrinsics(K2, size2, self.target_size)

        # 2. Compute dynamic rectification matrices
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=K1_scaled, distCoeffs1=D1,
            cameraMatrix2=K2_scaled, distCoeffs2=D2,
            imageSize=self.target_size,
            R=R, T=T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=self.target_size
        )

        # 3. Build single-pass remapping coordinate lookup tables
        map1_x, map1_y = cv2.initUndistortRectifyMap(K1_scaled, D1, R1, P1, self.target_size, cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(K2_scaled, D2, R2, P2, self.target_size, cv2.CV_32FC1)

        # 4. Execute single-pass remapping
        rectified_main = cv2.remap(img_main, map1_x, map1_y, cv2.INTER_LINEAR)
        rectified_uw = cv2.remap(img_uw, map2_x, map2_y, cv2.INTER_LINEAR)

        return rectified_main, rectified_uw, P1, P2, Q

    @staticmethod
    def disparity_to_depth(disparity: np.ndarray, f_rect: float, baseline: float, doffs: float = 0.0) -> np.ndarray:
        """
        Calculates metric depth Z = (f_rect * B) / (disparity + doffs).
        """
        valid_mask = disparity > 0
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[valid_mask] = (f_rect * baseline) / (disparity[valid_mask] + doffs)
        return depth
```

---

## Conclusion & Verification Summary

All mathematical transformations and algorithmic trade-offs for Issue #1 have been investigated and verified:
1. **FOV Alignment & Scaling Math**: Derived for crop offset, scaling factors, and intrinsic matrix update $K_2'$.
2. **OpenCV `stereoRectify` Internal Handling**: Single-pass unified remapping (`initUndistortRectifyMap` + `remap`) avoids pre-cropping blur and performs undistortion, FOV crop, alignment, and scale in one interpolation pass.
3. **Disparity-to-Depth Formula**: Scale-invariance under image downsampling validated.
