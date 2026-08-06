import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Point3D:
    """Structured 3D spatial coordinate."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_homogeneous(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, 1.0])


@dataclass(frozen=True)
class Vector3D:
    """Structured 3D direction vector."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class Cube3D:
    """Represents a 3D Cube geometry using homogeneous coordinates."""

    def __init__(self, size: float = 1.0):
        """
        Initialize a 3D cube centered at origin.
        
        :param size: Half-length of the cube side.
        """
        s = size
        # 8 vertices in homogeneous coordinates [X, Y, Z, 1]
        self.vertices = np.array([
            [-s, -s, -s, 1.0],  # 0: Back bottom-left
            [ s, -s, -s, 1.0],  # 1: Back bottom-right
            [ s,  s, -s, 1.0],  # 2: Back top-right
            [-s,  s, -s, 1.0],  # 3: Back top-left
            [-s, -s,  s, 1.0],  # 4: Front bottom-left
            [ s, -s,  s, 1.0],  # 5: Front bottom-right
            [ s,  s,  s, 1.0],  # 6: Front top-right
            [-s,  s,  s, 1.0]   # 7: Front top-left
        ])

        # 12 edges connecting pairs of vertex indices
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
