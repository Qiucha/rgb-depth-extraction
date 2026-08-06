"""
3D Cube Pinhole Camera & Stereo Room Projection Package with Integrated Depth Extraction Pipeline.
"""

from .geometry import Point3D, Vector3D, Cube3D
from .camera import PinholeCamera, StereoCameraRig, clip_polygon_near_plane
from .room_geometry import Face3D, RoomEnvironment, create_box_faces
from .controls import NavigationController
from .visualizer import CubeVisualizer
from .room_visualizer import RoomVisualizer
from .stereo_visualizer import StereoRoomVisualizer

__all__ = [
    "Point3D",
    "Vector3D",
    "Cube3D",
    "PinholeCamera",
    "StereoCameraRig",
    "clip_polygon_near_plane",
    "Face3D",
    "RoomEnvironment",
    "create_box_faces",
    "NavigationController",
    "CubeVisualizer",
    "RoomVisualizer",
    "StereoRoomVisualizer",
]
