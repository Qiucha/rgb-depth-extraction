# Robotics: RGB Camera Stereo Depth Extraction
> Created with Gemini & Codex

## Mission
Master the algorithmic mechanics of extracting 3D spatial depth maps from 2D stereo image pairs using epipolar block matching. This project explores similarity cost metrics (NCC vs. ZNCC), post-processing filters (WLS/Guided filtering), and the trade-offs between passive stereo matching and active structured light sensors in real-world robotics.

## Core Objectives
1. **Epipolar Matching:** Understand 1D scanline searching ($O(W \times H \times D)$) and cost volume computation.
2. **Depth Calculation:** Master the mathematical relation $Z = \frac{f \cdot B}{d + \text{doffs}}$ linking pixel shift to metric physical depth.
3. **Passive vs. Active Sensors:** Understand the fundamental trade-offs between passive optical stereo (multi-view matching) and active laboratory ground-truth capture (structured light pattern projection).

## Overview
This system estimates a **Disparity Map** from a rectified Stereo Pair and converts valid disparities into a metric **Depth Map**. It relies on epipolar sliding window block matching to locate corresponding features between left and right views along horizontal scanlines.

## Components
- **`digest/`**: Basic camera related material with interactive interface for intuitive learning.
- **`src/`**: Core logic for stereo depth extraction, including sliding window block matching and filtering.
- **`ios/`** & **`tools/`**: iOS dual-camera streaming and diagnostic tools.
- **`docs/`**: Specifications and documentation (e.g., iOS streaming diagnosis).
- **`tests/`**: Automated tests.
- **`run_pipeline.py` / `run_iphone_capture.py` / `serve_digest.py`**: Main entry points for processing and visualization.
