import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from typing import Optional, List, Tuple

from .camera import PinholeCamera, clip_polygon_near_plane
from .room_geometry import RoomEnvironment, Face3D
from .controls import NavigationController


class RoomVisualizer:
    """Renders a 3D Room Environment with near-plane polygon clipping, back-face culling, and camera-driven depth sorting."""

    WAYPOINTS = [
        ("1: Entrance View", (0.0, 0.0, -4.0)),
        ("2: Left Cube Focus", (-1.8, 0.2, 1.0)),
        ("3: Pedestal View", (0.0, 0.8, 2.5)),
        ("4: Right Pillar Focus", (1.8, 0.3, 3.8)),
        ("5: High Overview", (0.0, -1.5, 0.5)),
    ]

    def __init__(self, camera: PinholeCamera, room: RoomEnvironment, title: str = "3D Room Trajectory Visualizer"):
        self.camera = camera
        self.room = room
        self.title = title

        # Light source direction vector (top-left-front)
        self.light_dir = np.array([-0.4, -0.8, -0.5])

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.18)

        self.ax.set_xlim(0, self.camera.width)
        self.ax.set_ylim(self.camera.height, 0)  # Invert Y-axis for screen projection
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0f172a')  # Dark canvas background

        # Status text overlay (FOV & position)
        self.info_text = self.ax.text(
            0.02, 0.96, '', transform=self.ax.transAxes,
            fontsize=10, fontweight='bold', color='#38bdf8', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#334155', alpha=0.9)
        )

        # Controls legend overlay
        self.controls_text = self.ax.text(
            0.98, 0.96, '', transform=self.ax.transAxes,
            fontsize=8.5, color='#e2e8f0', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#334155', alpha=0.9)
        )

        self.slider = None
        self.anim = None
        self.is_animating = False

        self.initial_pos = (self.camera.pos_x, self.camera.pos_y, self.camera.pos_z)

        # Polygon patch container
        self.polygon_patches: List[Polygon] = []

    def update(self) -> None:
        """Render faces using near-plane clipping, back-face culling, and camera depth-only sorting."""
        for patch in self.polygon_patches:
            patch.remove()
        self.polygon_patches.clear()

        camera_pos = np.array([self.camera.pos_x, self.camera.pos_y, self.camera.pos_z])
        visible_faces = []
        for face in self.room.faces:
            if not face.is_visible_from(camera_pos):
                continue

            projected_2d, z_avg = self.camera.project_face(face, z_near=0.1)
            if projected_2d is not None and z_avg is not None:
                visible_faces.append((z_avg, face, projected_2d))

        visible_faces.sort(key=lambda item: item[0], reverse=True)

        for z_depth, face, projected_2d in visible_faces:
            shaded_color = face.compute_shaded_color(self.light_dir)
            poly = Polygon(
                projected_2d,
                closed=True,
                facecolor=shaded_color,
                edgecolor='#000000',
                linewidth=1.5,
                alpha=1.0
            )
            self.ax.add_patch(poly)
            self.polygon_patches.append(poly)

        fov_deg = self.camera.get_fov_degrees()
        pos_str = f"X: {self.camera.pos_x:+.2f}, Y: {self.camera.pos_y:+.2f}, Z: {self.camera.pos_z:+.2f}"
        self.info_text.set_text(
            f"Mode: Solid Opaque Room (Back-Face Culled + Depth Sorted)\n"
            f"Camera Pos: {pos_str} | FOV: {fov_deg:.1f}°\n"
            f"Faces Rendered: {len(visible_faces)} / {len(self.room.faces)}"
        )

        self.fig.canvas.draw_idle()

    def add_focal_length_slider(self, min_val: float = 200, max_val: float = 2000, step: float = 10) -> None:
        """Add focal length interactive slider."""
        ax_slider = plt.axes([0.15, 0.04, 0.45, 0.03], facecolor='#1e293b')
        self.slider = Slider(ax_slider, 'Focal Length', min_val, max_val, valinit=self.camera.focal_length, valstep=step, color='#38bdf8')

        def on_slider_change(val: float) -> None:
            self.camera.focal_length = val
            self.update()

        self.slider.on_changed(on_slider_change)

    def bind_controls(self, step_xy: float = 0.3, step_z: float = 0.3) -> None:
        """Bind keyboard controls using NavigationController."""
        legend = NavigationController.get_navigation_legend(allow_3d=True, include_waypoints=True)
        self.controls_text.set_text(legend)

        self.nav_controller = NavigationController(self.fig.canvas, self.update)

        def handle_room_keys(key: str) -> bool:
            if key in ('1', '2', '3', '4', '5'):
                self.jump_to_waypoint(int(key) - 1)
                return True
            elif key == 't':
                self.toggle_trajectory_tour()
                return True
            return False

        self.nav_controller.bind_controls(
            camera=self.camera,
            initial_pos=self.initial_pos,
            step_xy=step_xy,
            step_z=step_z,
            allow_3d=True,
            on_custom_key=handle_room_keys
        )

    def jump_to_waypoint(self, index: int) -> None:
        """Move camera directly to a trajectory waypoint."""
        if 0 <= index < len(self.WAYPOINTS):
            name, (x, y, z) = self.WAYPOINTS[index]
            self.camera.pos_x, self.camera.pos_y, self.camera.pos_z = x, y, z
            print(f"Jumped to Waypoint {index+1}: {name} -> ({x}, {y}, {z})")

    def toggle_trajectory_tour(self) -> None:
        """Toggle smooth trajectory tour animation across room waypoints."""
        if self.is_animating:
            if self.anim:
                self.anim.event_source.stop()
            self.is_animating = False
            print("Trajectory tour paused.")
        else:
            self.is_animating = True
            print("Trajectory tour started...")

            path_points = []
            num_points_between = 25
            coords = [wp[1] for wp in self.WAYPOINTS]
            coords.append(coords[0])

            for i in range(len(coords) - 1):
                p_start = np.array(coords[i])
                p_end = np.array(coords[i+1])
                for t in np.linspace(0, 1, num_points_between, endpoint=False):
                    interp = (1 - t) * p_start + t * p_end
                    path_points.append(interp)

            def animate_frame(frame_idx: int) -> None:
                if not self.is_animating:
                    return
                pt = path_points[frame_idx % len(path_points)]
                self.camera.pos_x, self.camera.pos_y, self.camera.pos_z = float(pt[0]), float(pt[1]), float(pt[2])
                self.update()

            self.anim = animation.FuncAnimation(self.fig, animate_frame, frames=len(path_points), interval=50, blit=False)
            self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Show interactive Matplotlib window."""
        self.update()
        plt.show()
