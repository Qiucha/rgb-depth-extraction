"""
Main entry point for running the 3D Room Stereo Camera Projection with Live Depth Information Extraction.

Displays a 2x2 Grid Window:
  - Top-Left: Left Camera View (im0)
  - Top-Right: Right Camera View (im1)
  - Bottom-Left: Overlapped Realtime Stereo Composite (Cyan / Rose Parallax Shift)
  - Bottom-Right: Live Extracted 3D Depth Map (Epipolar Block Matching & Depth Pipeline)
"""

import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath('.'))

from src.cube_projection import StereoCameraRig, RoomEnvironment, StereoRoomVisualizer


def main() -> None:
    """Run binocular stereo camera rig with live depth extraction pipeline."""
    # 1. Initialize Stereo Camera Rig (800px focal length, 0.2m baseline distance)
    stereo_rig = StereoCameraRig(
        focal_length=800.0,
        baseline=0.2,
        width=640,
        height=480,
        pos_x=0.0,
        pos_y=0.0,
        pos_z=-4.0
    )

    # 2. Build 3D Room Environment
    room = RoomEnvironment()

    # 3. Create Integrated Stereo Visualizer
    visualizer = StereoRoomVisualizer(
        stereo_rig=stereo_rig,
        room=room,
        title="Integrated Stereo 3D Room Visualizer & Live Depth Extraction Pipeline"
    )

    # 4. Add interactive sliders for Focal Length (f) and Baseline (B)
    visualizer.add_sliders(focal_range=(200, 2000, 10), baseline_range=(0.05, 1.0, 0.01))

    # 5. Bind keyboard controls (WASD/QE motion, 1-5 waypoints, T tour, P export)
    visualizer.bind_controls(step_xy=0.3, step_z=0.3)

    print("=== Launching Integrated 3D Room & Live Depth Extraction Window ===")
    print("Controls:")
    print("  W / S or ↑ / ↓ : Camera Up / Down")
    print("  A / D or ← / → : Camera Left / Right")
    print("  Q / E           : Camera Back / Forward")
    print("  1 - 5           : Jump to Waypoints")
    print("  T               : Toggle Trajectory Tour")
    print("  P               : Export Stereo Pair (im0.png, im1.png, calib.txt)")
    print("  R               : Reset Position")
    
    # 6. Display interactive window
    visualizer.show()


if __name__ == "__main__":
    main()
