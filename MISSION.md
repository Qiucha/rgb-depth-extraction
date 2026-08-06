# Mission: Mastering Stereo Disparity & 3D Spatial Depth Extraction

## User Goal
Master the algorithmic mechanics of extracting 3D spatial depth maps from 2D stereo image pairs using epipolar block matching, understanding similarity cost metrics (NCC vs. ZNCC), post-processing filters (WLS/Guided filtering), and why passive stereo matching is chosen over active ground-truth structured light sensors in real-world robotics.

## Core Objectives
1. Understand 1D epipolar scanline searching ($O(W \times H \times D)$) and cost volume computation.
2. Master the mathematical relation $Z = \frac{f \cdot B}{d + \text{doffs}}$ linking pixel shift to metric physical depth.
3. Understand the fundamental trade-off between passive optical stereo (passive multi-view matching) and active laboratory ground-truth capture (structured light pattern projection).
