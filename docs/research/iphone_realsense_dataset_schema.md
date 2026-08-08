# Research Report: Issue #2 - iPhone Dual-Cam & RealSense Benchmark Dataset Schema

## Executive Summary
This research investigation resolves **Issue #2** by defining a dual storage architecture (Unpacked Folder Hierarchy + JSON Manifest & Unified HDF5 Container) for recording and ingesting heterogeneous iPhone dual-camera streams alongside Intel RealSense reference ground-truth depth maps and Open3D 3D point clouds.

The schema handles per-frame dynamic camera calibration from iOS `AVCameraCalibrationData` ($K_1, K_2, R, T$, reference dimensions, distortion lookup tables), active IR reference depth from Intel RealSense D435/D455 (16-bit uint16 mm PNG / float32 meters `.npy`), and 3D point clouds (`.ply`/`.pcd`), while maintaining 100% backward compatibility with `src/stereo_depth` (Middlebury `calib.txt`, `im0.png`, `im1.png`, `disp0.pfm`).

---

## 1. Primary Source Analysis & Technical Standards

### A. Apple iOS `AVCameraCalibrationData` (AVFoundation)
* **Primary Source**: Apple Developer Documentation — `AVCameraCalibrationData`, `AVCaptureMultiCamSession`, `AVCaptureConnection.isCameraIntrinsicMatrixDeliveryEnabled`.
* **Dynamic Intrinsics ($K$)**: 
  $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
  Extracted per frame because Optical Image Stabilization (OIS) and voice-coil motor (VCM) autofocus shift the physical optics dynamically.
* **Reference Dimensions (`intrinsicMatrixReferenceDimensions`)**: `CGSize` (e.g., $4032 \times 3024$). Intrinsics scale linearly when images are resampled/cropped: $f_x' = f_x \cdot (w_{\text{new}} / w_{\text{ref}})$.
* **Extrinsics ($[R \mid T]$)**: Obtained via `AVCaptureDevice.extrinsicMatrix(from:to:)`. Transforms coordinates from Ultra-Wide camera frame to Main camera frame ($T_{\text{UW} \to \text{Main}}$).
* **Lens Distortion**: `lensDistortionLookupTable` (float array representing radial distortion profile) and `lensDistortionCenter` (optical axis offset relative to reference dimensions).

### B. Intel RealSense D435/D455 Active IR Benchmark Reference
* **Primary Source**: Intel RealSense SDK 2.0 (`librealsense2`) & D400 Series Datasheet.
* **Baseline ($B_{\text{RS}}$)**: $50\text{ mm}$ ($0.05\text{ m}$) on D435; $95\text{ mm}$ ($0.095\text{ m}$) on D455.
* **Reference Depth Map**:
  - **Z16 uint16 PNG**: Standard RealSense raw format where pixel value $V = \text{depth\_mm}$ (LSB = $0.001\text{ m}$). $Z_{\text{meters}} = V \times 0.001$.
  - **Float32 NumPy / PFM**: Direct metric depth map in meters, where invalid/unmapped depth values are stored as `0.0` or `NaN`/`inf`.
* **RealSense Intrinsics ($K_{\text{RS}}$)**: 3x3 matrix required for back-projecting depth pixels $(u, v, Z)$ into 3D point cloud coordinates $(X, Y, Z)^T$.

### C. Open3D 3D Point Cloud Dumps
* **Primary Source**: Open3D I/O API (`open3d.io.read_point_cloud`, `open3d.geometry.PointCloud`).
* **Formats**:
  - Binary `.ply` (Polygon File Format): Stores $(X, Y, Z)$ float32 coordinates and optional $(R, G, B)$ uint8 colors.
  - Binary `.pcd` (Point Cloud Data): Standard PCL format supported by Open3D.

---

## 2. Folder Hierarchy Specification (Unpacked Directory Storage)

```
data/realworld/
└── sequence_001_office_desk/
    ├── dataset_manifest.json            # Global sequence metadata & full frame index
    ├── adapter_calib.txt                # Middlebury-compatible synthesized calib.txt
    ├── frame_000000/
    │   ├── frame_meta.json              # Detailed per-frame calibration & timestamps
    │   ├── im0_main.png                 # Main camera RGB (e.g. 4032x3024 or 1920x1080)
    │   ├── im1_ultrawide.png            # Ultra-Wide camera RGB (e.g. 4032x3024 or 1920x1080)
    │   ├── realsense_depth.png          # RealSense ground-truth depth (16-bit uint16 PNG, unit: mm)
    │   ├── realsense_depth.npy          # RealSense metric depth map (float32 array, unit: meters)
    │   ├── realsense_gt.ply             # Open3D 3D point cloud ground truth dump (binary PLY)
    │   └── realsense_gt.pcd             # Open3D 3D point cloud ground truth dump (binary PCD)
    ├── frame_000001/
    │   └── ...
    └── ...
```

---

## 3. Formal JSON Schema (`dataset_manifest.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "iPhoneRealSenseDatasetManifest",
  "description": "Schema for iPhone dual-camera & Intel RealSense benchmark dataset captures",
  "type": "object",
  "required": ["sequence_id", "version", "created_at", "devices", "frame_count", "frames"],
  "properties": {
    "sequence_id": { "type": "string", "example": "seq_2026_08_08_001" },
    "version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$", "default": "1.0.0" },
    "created_at": { "type": "string", "format": "date-time" },
    "description": { "type": "string" },
    "environment": {
      "type": "object",
      "properties": {
        "lighting_condition": { "type": "string", "enum": ["bright_indoor", "dim_indoor", "outdoor_sun", "outdoor_shade", "mixed"] },
        "surface_texture_type": { "type": "string", "enum": ["textured", "textureless", "mixed", "specular"] },
        "nominal_distance_meters": { "type": "number", "minimum": 0.1 }
      }
    },
    "devices": {
      "type": "object",
      "required": ["iphone", "realsense"],
      "properties": {
        "iphone": {
          "type": "object",
          "required": ["model", "os_version", "nominal_baseline_mm", "main_camera", "ultrawide_camera"],
          "properties": {
            "model": { "type": "string", "example": "iPhone 15 Pro" },
            "os_version": { "type": "string", "example": "iOS 17.4" },
            "nominal_baseline_mm": { "type": "number", "example": 19.5 },
            "main_camera": {
              "type": "object",
              "required": ["sensor_name", "nominal_focal_length_mm", "nominal_fov_deg"],
              "properties": {
                "sensor_name": { "type": "string", "example": "Main Wide 24mm equivalent" },
                "nominal_focal_length_mm": { "type": "number", "example": 6.86 },
                "nominal_fov_deg": { "type": "number", "example": 80.0 }
              }
            },
            "ultrawide_camera": {
              "type": "object",
              "required": ["sensor_name", "nominal_focal_length_mm", "nominal_fov_deg"],
              "properties": {
                "sensor_name": { "type": "string", "example": "Ultra-Wide 13mm equivalent" },
                "nominal_focal_length_mm": { "type": "number", "example": 2.22 },
                "nominal_fov_deg": { "type": "number", "example": 120.0 }
              }
            }
          }
        },
        "realsense": {
          "type": "object",
          "required": ["model", "serial_number", "firmware_version", "baseline_mm", "depth_units_meters"],
          "properties": {
            "model": { "type": "string", "enum": ["D435", "D435i", "D455", "D415"] },
            "serial_number": { "type": "string" },
            "firmware_version": { "type": "string" },
            "baseline_mm": { "type": "number", "example": 50.0 },
            "depth_units_meters": { "type": "number", "default": 0.001 },
            "emitter_enabled": { "type": "boolean", "default": true }
          }
        }
      }
    },
    "frame_count": { "type": "integer", "minimum": 1 },
    "frames": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "frame_index",
          "timestamp_seconds",
          "files",
          "iphone_calibration",
          "realsense_calibration"
        ],
        "properties": {
          "frame_index": { "type": "integer", "minimum": 0 },
          "timestamp_seconds": { "type": "number" },
          "files": {
            "type": "object",
            "required": ["main_rgb", "ultrawide_rgb", "realsense_depth_png", "realsense_depth_npy", "realsense_pointcloud_ply"],
            "properties": {
              "main_rgb": { "type": "string", "example": "frame_000000/im0_main.png" },
              "ultrawide_rgb": { "type": "string", "example": "frame_000000/im1_ultrawide.png" },
              "realsense_depth_png": { "type": "string", "example": "frame_000000/realsense_depth.png" },
              "realsense_depth_npy": { "type": "string", "example": "frame_000000/realsense_depth.npy" },
              "realsense_pointcloud_ply": { "type": "string", "example": "frame_000000/realsense_gt.ply" },
              "realsense_pointcloud_pcd": { "type": "string", "example": "frame_000000/realsense_gt.pcd" }
            }
          },
          "iphone_calibration": {
            "type": "object",
            "required": [
              "reference_dimensions",
              "main_intrinsics",
              "ultrawide_intrinsics",
              "extrinsic_transform_ultrawide_to_main"
            ],
            "properties": {
              "reference_dimensions": {
                "type": "object",
                "required": ["width", "height"],
                "properties": {
                  "width": { "type": "integer", "example": 4032 },
                  "height": { "type": "integer", "example": 3024 }
                }
              },
              "main_intrinsics": {
                "type": "object",
                "required": ["matrix_3x3", "pixel_size_micrometers", "lens_position"],
                "properties": {
                  "matrix_3x3": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } }
                  },
                  "pixel_size_micrometers": { "type": "number", "example": 1.22 },
                  "lens_position": { "type": "number", "example": 0.45 },
                  "distortion_center": { "type": "array", "items": { "type": "number" } },
                  "distortion_lookup_table": { "type": "array", "items": { "type": "number" } }
                }
              },
              "ultrawide_intrinsics": {
                "type": "object",
                "required": ["matrix_3x3", "pixel_size_micrometers", "lens_position"],
                "properties": {
                  "matrix_3x3": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } }
                  },
                  "pixel_size_micrometers": { "type": "number", "example": 1.0 },
                  "lens_position": { "type": "number", "example": 0.0 },
                  "distortion_center": { "type": "array", "items": { "type": "number" } },
                  "distortion_lookup_table": { "type": "array", "items": { "type": "number" } }
                }
              },
              "extrinsic_transform_ultrawide_to_main": {
                "type": "object",
                "required": ["rotation_matrix_3x3", "translation_vector_mm"],
                "properties": {
                  "rotation_matrix_3x3": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } }
                  },
                  "translation_vector_mm": {
                    "type": "array",
                    "items": { "type": "number" },
                    "minItems": 3,
                    "maxItems": 3
                  }
                }
              }
            }
          },
          "realsense_calibration": {
            "type": "object",
            "required": ["dimensions", "intrinsics_matrix_3x3", "depth_scale_meters"],
            "properties": {
              "dimensions": {
                "type": "object",
                "required": ["width", "height"],
                "properties": {
                  "width": { "type": "integer", "example": 1280 },
                  "height": { "type": "integer", "example": 720 }
                }
              },
              "intrinsics_matrix_3x3": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } }
              },
              "depth_scale_meters": { "type": "number", "example": 0.001 }
            }
          }
        }
      }
    }
  }
}
```

---

## 4. Unified HDF5 Binary Container Architecture (`sequence_001.h5`)

For high-throughput Python loading via `h5py`, the dataset can also be packaged as a single HDF5 container:

```
/ (Root Attributes: sequence_id, version, created_at, iphone_model, realsense_model, baseline_mm)
│
├── /frame_000000/
│   ├── rgb_main                       [Dataset: uint8, shape (H1, W1, 3), chunked, gzip level 4]
│   ├── rgb_ultrawide                  [Dataset: uint8, shape (H2, W2, 3), chunked, gzip level 4]
│   ├── realsense_depth                [Dataset: float32, shape (H_rs, W_rs), unit: meters]
│   ├── realsense_pointcloud/
│   │   ├── points                     [Dataset: float32, shape (N, 3), unit: meters]
│   │   └── colors                     [Dataset: uint8, shape (N, 3), range [0, 255]]
│   └── calibration/
│       ├── K_main                     [Dataset: float32, shape (3, 3)]
│       ├── K_ultrawide                [Dataset: float32, shape (3, 3)]
│       ├── R_uw_to_main               [Dataset: float32, shape (3, 3)]
│       ├── T_uw_to_main               [Dataset: float32, shape (3, 1), unit: mm]
│       ├── K_realsense                [Dataset: float32, shape (3, 3)]
│       └── distortion_lut_main        [Dataset: float32, shape (L,)]
└── /frame_000001/
    └── ...
```

---

## 5. Middlebury Compatibility Layer (`src/stereo_depth` Integration)

To allow `src/stereo_depth/` (`SlidingWindowMatcher`, `DepthCalculator`, `DepthEvaluator`) to process realworld captures without modifying core modules, we define a synthesized `calib.txt` adapter function:

```python
def export_to_middlebury_format(frame_meta: dict, output_dir: str):
    """
    Converts per-frame dynamic calibration from frame_meta into Middlebury calib.txt:
    cam0 = [f_x 0 c_x; 0 f_y c_y; 0 0 1]
    cam1 = [f_x 0 c_x; 0 f_y c_y; 0 0 1]
    doffs = 0.0
    baseline = baseline_mm / 1000.0
    width = W
    height = H
    ndisp = 128
    """
    K_main = frame_meta["iphone_calibration"]["main_intrinsics"]["matrix_3x3"]
    T = frame_meta["iphone_calibration"]["extrinsic_transform_ultrawide_to_main"]["translation_vector_mm"]
    baseline_mm = math.sqrt(T[0]**2 + T[1]**2 + T[2]**2)

    cam0_str = f"[{K_main[0][0]:.4f} 0 {K_main[0][2]:.4f}; 0 {K_main[1][1]:.4f} {K_main[1][2]:.4f}; 0 0 1]"
    
    calib_content = (
        f"cam0={cam0_str}\n"
        f"cam1={cam0_str}\n"
        f"doffs=0.0\n"
        f"baseline={baseline_mm:.4f}\n"
        f"width={frame_meta['iphone_calibration']['reference_dimensions']['width']}\n"
        f"height={frame_meta['iphone_calibration']['reference_dimensions']['height']}\n"
        f"ndisp=128\n"
        f"vmin=0\n"
        f"vmax=255\n"
    )
    with open(os.path.join(output_dir, "calib.txt"), "w") as f:
        f.write(calib_content)
```
