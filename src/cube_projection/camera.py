import numpy as np
from typing import Tuple, Optional, Union

from .geometry import Point3D, Vector3D


def clip_polygon_near_plane(vertices_cam: np.ndarray, z_near: float = 0.1) -> Optional[np.ndarray]:
    """
    Sutherland-Hodgman polygon clipping algorithm against the camera near-plane (Z_cam >= z_near).
    Prevents vertices behind the camera plane from projecting inverted coordinates onto the screen.
    """
    out_verts = []
    n = len(vertices_cam)
    for i in range(n):
        curr = vertices_cam[i]
        prev = vertices_cam[i - 1]
        curr_inside = (curr[2] >= z_near)
        prev_inside = (prev[2] >= z_near)

        if curr_inside:
            if not prev_inside:
                t = (z_near - prev[2]) / (curr[2] - prev[2])
                inter = prev + t * (curr - prev)
                inter[2] = z_near
                out_verts.append(inter)
            out_verts.append(curr)
        elif prev_inside:
            t = (z_near - prev[2]) / (curr[2] - prev[2])
            inter = prev + t * (curr - prev)
            inter[2] = z_near
            out_verts.append(inter)

    return np.array(out_verts, dtype=np.float64) if len(out_verts) >= 3 else None


class PinholeCamera:
    """Simulates a Pinhole Camera model with intrinsics and extrinsics."""

    def __init__(self, focal_length: float = 800.0, width: int = 640, height: int = 480,
                 pos_x: float = 0.0, pos_y: float = 0.0, pos_z: float = -5.0,
                 position: Optional[Point3D] = None):
        self.focal_length = focal_length
        self.width = width
        self.height = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.position = position if position is not None else Point3D(pos_x, pos_y, pos_z)

    @property
    def pos_x(self) -> float:
        return self.position.x

    @pos_x.setter
    def pos_x(self, val: float) -> None:
        self.position = Point3D(val, self.position.y, self.position.z)

    @property
    def pos_y(self) -> float:
        return self.position.y

    @pos_y.setter
    def pos_y(self, val: float) -> None:
        self.position = Point3D(self.position.x, val, self.position.z)

    @property
    def pos_z(self) -> float:
        return self.position.z

    @pos_z.setter
    def pos_z(self, val: float) -> None:
        self.position = Point3D(self.position.x, self.position.y, val)

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Translate camera position by (dx, dy, dz)."""
        self.position = Point3D(self.position.x + dx, self.position.y + dy, self.position.z + dz)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Construct 3x3 Camera Intrinsic Matrix M_int."""
        return np.array([
            [self.focal_length, 0, self.cx],
            [0, self.focal_length, self.cy],
            [0, 0, 1.0]
        ])

    @property
    def homogeneous_intrinsic_matrix(self) -> np.ndarray:
        """Construct 3x4 Homogeneous Intrinsic Matrix K_3x4 = [K_3x3 | 0]."""
        return np.hstack((self.intrinsic_matrix, np.zeros((3, 1))))

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """Construct 3x4 Camera Extrinsic Matrix M_ext (Translation vector t, Identity R)."""
        R = np.eye(3)
        t = np.array([[-self.pos_x], [-self.pos_y], [-self.pos_z]])
        return np.hstack((R, t))

    @property
    def projection_matrix(self) -> np.ndarray:
        """Compute 3x4 Camera Projection Matrix P = M_int @ M_ext."""
        return self.intrinsic_matrix @ self.extrinsic_matrix

    def to_camera_space(self, vertices: np.ndarray) -> np.ndarray:
        """Transform 3D homogeneous world vertices (N x 4) into 3D camera-space coordinates (N x 3)."""
        return vertices @ self.extrinsic_matrix.T

    def _divide_perspective(self, projected_homo: np.ndarray) -> np.ndarray:
        """Perform perspective division on homogeneous projected coordinates."""
        z_vals = projected_homo[:, 2:]
        return projected_homo[:, :2] / z_vals

    def project_face(self, face, z_near: float = 0.1) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Deep interface method: Transform, clip, and project a Face3D polygon onto 2D image plane.
        
        :param face: Face3D instance with 3D homogeneous vertices.
        :param z_near: Near-plane clipping threshold.
        :return: Tuple of (projected_2d (N x 2), z_avg_depth) or (None, None).
        """
        vertices_cam = self.to_camera_space(face.vertices)
        clipped_cam = clip_polygon_near_plane(vertices_cam, z_near=z_near)
        if clipped_cam is None:
            return None, None
        projected_homo = clipped_cam @ self.intrinsic_matrix.T
        projected_2d = self._divide_perspective(projected_homo)
        z_avg = float(np.mean(clipped_cam[:, 2]))
        return projected_2d, z_avg

    def project_vertices(self, vertices: np.ndarray, return_depth: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Project 3D homogeneous vertices (N x 4) onto 2D image plane (N x 2).
        
        :param vertices: Homogeneous 3D vertices array of shape (N, 4).
        :param return_depth: If True, returns tuple of (projected_2d, z_depths).
        :return: 2D pixel coordinates of shape (N, 2), or tuple with z_depths.
        """
        P = self.projection_matrix
        projected_homo = vertices @ P.T
        projected_2d = self._divide_perspective(projected_homo)
        if return_depth:
            z_depths = projected_homo[:, 2]
            return projected_2d, z_depths
        return projected_2d

    def get_fov_degrees(self) -> float:
        """Calculate Horizontal Field of View (FOV) in degrees."""
        fov_rad = 2.0 * np.arctan((self.width / 2.0) / self.focal_length)
        return float(np.degrees(fov_rad))


class StereoCameraRig:
    """
    Simulates a binocular Stereo Camera Rig with Left (C_L) and Right (C_R) pinhole cameras
    horizontally displaced by baseline distance B (baseline = distance between camera centers).
    
    Center position (X, Y, Z):
      Left camera position:  (X - B/2, Y, Z)
      Right camera position: (X + B/2, Y, Z)
    """

    def __init__(self, focal_length: float = 800.0, baseline: float = 0.2,
                 width: int = 640, height: int = 480,
                 pos_x: float = 0.0, pos_y: float = 0.0, pos_z: float = -4.0):
        self._baseline = baseline
        self._pos_x = pos_x
        self._pos_y = pos_y
        self._pos_z = pos_z
        self._focal_length = focal_length
        self._width = width
        self._height = height

        self.left_camera = PinholeCamera(
            focal_length=focal_length, width=width, height=height,
            pos_x=pos_x - baseline / 2.0, pos_y=pos_y, pos_z=pos_z
        )
        self.right_camera = PinholeCamera(
            focal_length=focal_length, width=width, height=height,
            pos_x=pos_x + baseline / 2.0, pos_y=pos_y, pos_z=pos_z
        )

    def _sync_cameras(self) -> None:
        self.left_camera.focal_length = self._focal_length
        self.left_camera.width = self._width
        self.left_camera.height = self._height
        self.left_camera.pos_x = self._pos_x - self._baseline / 2.0
        self.left_camera.pos_y = self._pos_y
        self.left_camera.pos_z = self._pos_z

        self.right_camera.focal_length = self._focal_length
        self.right_camera.width = self._width
        self.right_camera.height = self._height
        self.right_camera.pos_x = self._pos_x + self._baseline / 2.0
        self.right_camera.pos_y = self._pos_y
        self.right_camera.pos_z = self._pos_z

    @property
    def baseline(self) -> float:
        return self._baseline

    @baseline.setter
    def baseline(self, val: float) -> None:
        self._baseline = float(val)
        self._sync_cameras()

    @property
    def focal_length(self) -> float:
        return self._focal_length

    @focal_length.setter
    def focal_length(self, val: float) -> None:
        self._focal_length = float(val)
        self._sync_cameras()

    @property
    def pos_x(self) -> float:
        return self._pos_x

    @pos_x.setter
    def pos_x(self, val: float) -> None:
        self._pos_x = float(val)
        self._sync_cameras()

    @property
    def pos_y(self) -> float:
        return self._pos_y

    @pos_y.setter
    def pos_y(self, val: float) -> None:
        self._pos_y = float(val)
        self._sync_cameras()

    @property
    def pos_z(self) -> float:
        return self._pos_z

    @pos_z.setter
    def pos_z(self, val: float) -> None:
        self._pos_z = float(val)
        self._sync_cameras()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Translate stereo camera rig center by (dx, dy, dz)."""
        self._pos_x += dx
        self._pos_y += dy
        self._pos_z += dz
        self._sync_cameras()

    def get_fov_degrees(self) -> float:
        return self.left_camera.get_fov_degrees()
