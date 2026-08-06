import numpy as np
import matplotlib.colors as mcolors
from typing import List, Tuple, Union, Optional
from .geometry import Vector3D, Point3D


class Face3D:
    """Represents a 3D planar face (polygon) with surface normal and solid base color."""

    def __init__(self, vertices: np.ndarray, base_color: str, normal: Union[Vector3D, np.ndarray], label: str = ""):
        """
        :param vertices: Array of shape (N, 4) in homogeneous coordinates [X, Y, Z, 1.0].
        :param base_color: Hex color string or RGB tuple.
        :param normal: 3D normal vector (Vector3D or numpy array [nx, ny, nz]).
        :param label: Descriptive label for the face/object.
        """
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.base_color = base_color
        raw_norm = normal.to_array() if isinstance(normal, Vector3D) else np.array(normal, dtype=np.float64)
        norm_val = np.linalg.norm(raw_norm)
        self.normal = (raw_norm / norm_val) if norm_val > 0 else raw_norm
        self.label = label

    @property
    def center(self) -> np.ndarray:
        """Compute the 3D centroid of the face in world coordinates [X, Y, Z]."""
        return np.mean(self.vertices[:, :3], axis=0)

    def is_visible_from(self, camera_pos: np.ndarray) -> bool:
        """
        Check if face is front-facing (visible) relative to camera position in world space.
        
        :param camera_pos: 3D camera position [X, Y, Z] in world space.
        :return: True if front-facing towards camera, False if back-facing (culled).
        """
        view_vector = self.center - camera_pos[:3]
        return float(np.dot(self.normal, view_vector)) < 0.0

    def compute_shaded_color(self, light_dir: np.ndarray, ambient: float = 0.35, diffuse_weight: float = 0.65) -> Tuple[float, float, float]:
        """
        Compute directional light shading (Lambertian reflectance) for solid opaque rendering.
        
        :param light_dir: Direction vector towards light source (unit vector).
        :param ambient: Minimum ambient light intensity.
        :param diffuse_weight: Weight for directional diffuse component.
        :return: (r, g, b) tuple scaled by shading intensity.
        """
        light_unit = light_dir / np.linalg.norm(light_dir)
        cos_theta = max(0.0, float(np.dot(self.normal, -light_unit)))
        intensity = ambient + diffuse_weight * cos_theta
        intensity = min(1.0, max(0.0, intensity))

        base_rgb = mcolors.to_rgb(self.base_color)
        return (base_rgb[0] * intensity, base_rgb[1] * intensity, base_rgb[2] * intensity)


def create_box_faces(cx: float, cy: float, cz: float,
                     half_x: float, half_y: float, half_z: float,
                     base_color: str, label: str = "") -> List[Face3D]:
    """
    Create 6 Quad Face3D instances for an axis-aligned 3D cuboid centered at (cx, cy, cz).
    
    :param cx, cy, cz: Center position in 3D world coordinates.
    :param half_x, half_y, half_z: Half-lengths along X, Y, Z axes.
    :param base_color: Base hex color string for the box.
    :param label: Object label.
    :return: List of 6 Face3D instances with outward-facing normals.
    """
    v = np.array([
        [cx - half_x, cy - half_y, cz - half_z, 1.0],  # 0: Back top-left
        [cx + half_x, cy - half_y, cz - half_z, 1.0],  # 1: Back top-right
        [cx + half_x, cy + half_y, cz - half_z, 1.0],  # 2: Back bottom-right
        [cx - half_x, cy + half_y, cz - half_z, 1.0],  # 3: Back bottom-left
        [cx - half_x, cy - half_y, cz + half_z, 1.0],  # 4: Front top-left
        [cx + half_x, cy - half_y, cz + half_z, 1.0],  # 5: Front top-right
        [cx + half_x, cy + half_y, cz + half_z, 1.0],  # 6: Front bottom-right
        [cx - half_x, cy + half_y, cz + half_z, 1.0]   # 7: Front bottom-left
    ])

    faces = [
        # Front Face (+Z)
        Face3D(v[[4, 5, 6, 7]], base_color, np.array([0.0, 0.0, 1.0]), f"{label}_front"),
        # Back Face (-Z)
        Face3D(v[[1, 0, 3, 2]], base_color, np.array([0.0, 0.0, -1.0]), f"{label}_back"),
        # Top Face (-Y)
        Face3D(v[[0, 1, 5, 4]], base_color, np.array([0.0, -1.0, 0.0]), f"{label}_top"),
        # Bottom Face (+Y)
        Face3D(v[[7, 6, 2, 3]], base_color, np.array([0.0, 1.0, 0.0]), f"{label}_bottom"),
        # Left Face (-X)
        Face3D(v[[0, 4, 7, 3]], base_color, np.array([-1.0, 0.0, 0.0]), f"{label}_left"),
        # Right Face (+X)
        Face3D(v[[5, 1, 2, 6]], base_color, np.array([1.0, 0.0, 0.0]), f"{label}_right"),
    ]
    return faces


class RoomEnvironment:
    """Builds a solid 5-sided room container with multiple colored 3D objects inside."""

    def __init__(self) -> None:
        self.faces: List[Face3D] = []
        self._build_room()
        self._build_objects()

    def _build_room(self) -> None:
        """Construct the 5 room enclosure surfaces (Floor, Ceiling, Back Wall, Left Wall, Right Wall)."""
        x_min, x_max = -4.0, 4.0
        y_min, y_max = -2.5, 2.5
        z_min, z_max = 0.0, 10.0

        # Floor (Y = y_max, facing up [-Y]) -> Base color: Slate Gray #64748b
        v_floor = np.array([
            [x_min, y_max, z_min, 1.0],
            [x_max, y_max, z_min, 1.0],
            [x_max, y_max, z_max, 1.0],
            [x_min, y_max, z_max, 1.0]
        ])
        self.faces.append(Face3D(v_floor, "#64748b", np.array([0.0, -1.0, 0.0]), "room_floor"))

        # Ceiling (Y = y_min, facing down [+Y]) -> Base color: Soft Off-White #e2e8f0
        v_ceil = np.array([
            [x_min, y_min, z_max, 1.0],
            [x_max, y_min, z_max, 1.0],
            [x_max, y_min, z_min, 1.0],
            [x_min, y_min, z_min, 1.0]
        ])
        self.faces.append(Face3D(v_ceil, "#e2e8f0", np.array([0.0, 1.0, 0.0]), "room_ceiling"))

        # Back Wall (Z = z_max, facing forward [-Z]) -> Base color: Deep Indigo/Navy #1e293b
        v_back = np.array([
            [x_min, y_min, z_max, 1.0],
            [x_max, y_min, z_max, 1.0],
            [x_max, y_max, z_max, 1.0],
            [x_min, y_max, z_max, 1.0]
        ])
        self.faces.append(Face3D(v_back, "#1e293b", np.array([0.0, 0.0, -1.0]), "room_back_wall"))

        # Left Wall (X = x_min, facing right [+X]) -> Base color: Warm Amber/Terracotta #d97706
        v_left = np.array([
            [x_min, y_min, z_min, 1.0],
            [x_min, y_min, z_max, 1.0],
            [x_min, y_max, z_max, 1.0],
            [x_min, y_max, z_min, 1.0]
        ])
        self.faces.append(Face3D(v_left, "#d97706", np.array([1.0, 0.0, 0.0]), "room_left_wall"))

        # Right Wall (X = x_max, facing left [-X]) -> Base color: Forest Emerald Green #059669
        v_right = np.array([
            [x_max, y_min, z_max, 1.0],
            [x_max, y_min, z_min, 1.0],
            [x_max, y_max, z_min, 1.0],
            [x_max, y_max, z_max, 1.0]
        ])
        self.faces.append(Face3D(v_right, "#059669", np.array([-1.0, 0.0, 0.0]), "room_right_wall"))

    def _build_objects(self) -> None:
        """Construct multiple distinct 3D shapes inside the room."""
        # 1. Central Pedestal / Table (Solid Crimson Red #dc2626)
        self.faces.extend(create_box_faces(cx=0.0, cy=1.7, cz=5.0, half_x=0.8, half_y=0.4, half_z=0.8, base_color="#dc2626", label="pedestal"))

        # 2. Left Floating Cube (Solid Cyan/Teal #06b6d4)
        self.faces.extend(create_box_faces(cx=-2.0, cy=0.5, cz=3.5, half_x=0.5, half_y=0.5, half_z=0.5, base_color="#06b6d4", label="left_cube"))

        # 3. Right Tall Pillar (Solid Gold/Amber #f59e0b)
        self.faces.extend(create_box_faces(cx=2.2, cy=0.8, cz=6.5, half_x=0.45, half_y=0.9, half_z=0.45, base_color="#f59e0b", label="right_pillar"))

        # 4. Back Floating Small Cube (Solid Royal Purple #8b5cf6)
        self.faces.extend(create_box_faces(cx=-1.2, cy=-0.8, cz=8.0, half_x=0.4, half_y=0.4, half_z=0.4, base_color="#8b5cf6", label="back_cube"))

        # 5. Background Monolith Block (Solid Coral Pink #ec4899)
        self.faces.extend(create_box_faces(cx=0.4, cy=0.8, cz=7.2, half_x=0.8, half_y=1.0, half_z=0.5, base_color="#ec4899", label="back_monolith"))
