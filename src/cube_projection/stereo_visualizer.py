import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from typing import Optional, List, Tuple

from .camera import PinholeCamera, StereoCameraRig
from .room_geometry import RoomEnvironment, Face3D
from .controls import NavigationController

from src.stereo_depth.sliding_window import SlidingWindowMatcher
from src.stereo_depth.depth_calculator import DepthCalculator


class StereoRoomVisualizer:
    """
    Renders a 3D Room Environment simultaneously from Left (im0) and Right (im1) virtual pinhole cameras
    (Stereo Camera Rig) in a 2x2 grid layout:
      - Top-Left: Left Camera View (im0)
      - Top-Right: Right Camera View (im1)
      - Bottom-Left: Overlapped Realtime Stereo Composite (Parallax Shift Overlay)
      - Bottom-Right: Integrated Depth Extraction Pipeline Map (Direct Epipolar Matching & Depth Calc)
    """

    WAYPOINTS = [
        ("1: Entrance View", (0.0, 0.0, -4.0)),
        ("2: Left Cube Focus", (-1.8, 0.2, 1.0)),
        ("3: Pedestal View", (0.0, 0.8, 2.5)),
        ("4: Right Pillar Focus", (1.8, 0.3, 3.8)),
        ("5: High Overview", (0.0, -1.5, 0.5)),
    ]

    def __init__(self, stereo_rig: StereoCameraRig, room: RoomEnvironment, title: str = "Stereo 3D Room Visualizer with Direct Depth Extraction Pipeline", enable_dots: bool = False):
        """
        Initialize StereoRoomVisualizer with injected stereo camera rig, room environment, and optional dot grid texture rendering.
        
        :param stereo_rig: StereoCameraRig instance.
        :param room: RoomEnvironment instance.
        :param title: Window title.
        :param enable_dots: If True, renders high-contrast dot grids on face surfaces for accurate passive stereo depth extraction.
        """
        self.stereo_rig = stereo_rig
        self.room = room
        self.title = title
        self.enable_dots = enable_dots

        # Light source direction vector
        self.light_dir = np.array([-0.4, -0.8, -0.5])

        # Initialize matcher & depth calculator pipeline
        self.matcher = SlidingWindowMatcher(window_size=7, max_disparity=48, min_disparity=0, metric='ncc')
        self.depth_calculator = DepthCalculator(
            focal_length=self.stereo_rig.focal_length,
            baseline=self.stereo_rig.baseline,
            doffs=0.0,
            cx=self.stereo_rig.left_camera.cx,
            cy=self.stereo_rig.left_camera.cy
        )

        # 2x2 Grid Layout
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
        self.ax_left = axes[0, 0]
        self.ax_right = axes[0, 1]
        self.ax_overlap = axes[1, 0]
        self.ax_depth = axes[1, 1]

        plt.subplots_adjust(bottom=0.12, top=0.93, hspace=0.25, wspace=0.15)
        self.fig.suptitle(title, fontsize=14, fontweight='bold', color='#0f172a')

        # Active camera view axes setup
        active_axes = [
            (self.ax_left, "Left View (im0)"),
            (self.ax_right, "Right View (im1)"),
            (self.ax_overlap, "Overlapped Stereo Composite (Left: Cyan / Right: Rose)")
        ]
        for ax, label in active_axes:
            ax.set_xlim(0, self.stereo_rig.width)
            ax.set_ylim(self.stereo_rig.height, 0)  # Invert Y-axis for screen projection
            ax.set_aspect('equal')
            ax.set_facecolor('#0f172a')  # Dark canvas background
            ax.set_title(label, fontsize=10.5, fontweight='bold', color='#3b82f6', pad=6)

        # Bottom-Right quadrant for extracted depth map
        self.ax_depth.set_facecolor('#090d16')
        self.ax_depth.set_title("Extracted 3D Depth Map (Direct Epipolar Matching Pipeline)", fontsize=10.5, fontweight='bold', color='#10b981', pad=6)
        self.depth_im = None
        self.depth_cbar = None

        # Status text overlay (on Top-Left)
        self.info_text = self.ax_left.text(
            0.02, 0.96, '', transform=self.ax_left.transAxes,
            fontsize=8.5, fontweight='bold', color='#38bdf8', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#334155', alpha=0.9)
        )

        # Controls legend overlay (on Top-Right)
        self.controls_text = self.ax_right.text(
            0.98, 0.96, '', transform=self.ax_right.transAxes,
            fontsize=8, color='#e2e8f0', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#334155', alpha=0.9)
        )

        self.slider_focal = None
        self.slider_baseline = None
        self.anim = None
        self.is_animating = False

        self.initial_pos = (self.stereo_rig.pos_x, self.stereo_rig.pos_y, self.stereo_rig.pos_z)

        # Patch containers
        self.left_patches: List[Polygon] = []
        self.right_patches: List[Polygon] = []
        self.overlap_patches: List[Polygon] = []
        self.dot_artists: List[object] = []

    def _render_camera_view(self, camera: PinholeCamera, ax: plt.Axes, patch_list: List[Polygon]) -> List[Tuple[float, Face3D, np.ndarray]]:
        """Render room geometry onto target axis using specified camera model."""
        for patch in patch_list:
            patch.remove()
        patch_list.clear()

        camera_pos = np.array([camera.pos_x, camera.pos_y, camera.pos_z])
        visible_faces = []
        for face in self.room.faces:
            if not face.is_visible_from(camera_pos):
                continue
            projected_2d, z_avg = camera.project_face(face, z_near=0.1)
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
                linewidth=1.1,
                alpha=1.0
            )
            ax.add_patch(poly)
            patch_list.append(poly)

            # Draw surface dot grid textures if enabled
            if self.enable_dots:
                dots_3d = face.generate_surface_dots(grid_res=6)
                if len(dots_3d) > 0:
                    dots_2d, z_depths = camera.project_vertices(dots_3d, return_depth=True)
                    valid_dots = (z_depths > 0.1) & \
                                 (dots_2d[:, 0] >= 0) & (dots_2d[:, 0] <= camera.width) & \
                                 (dots_2d[:, 1] >= 0) & (dots_2d[:, 1] <= camera.height)
                    if np.any(valid_dots):
                        pts = dots_2d[valid_dots]
                        line, = ax.plot(pts[:, 0], pts[:, 1], 'o', color='#ffffff', markeredgecolor='#000000', markeredgewidth=0.5, markersize=2.5, zorder=5)
                        self.dot_artists.append(line)

        return visible_faces

    def _render_overlapped_view(self, left_faces: List[Tuple[float, Face3D, np.ndarray]], right_faces: List[Tuple[float, Face3D, np.ndarray]]) -> None:
        """Render real-time overlapped composite view (Left in Cyan, Right in Rose) in bottom-left subplot."""
        for patch in self.overlap_patches:
            patch.remove()
        self.overlap_patches.clear()

        for z_depth, face, projected_2d in left_faces:
            poly_left = Polygon(
                projected_2d,
                closed=True,
                facecolor='#38bdf8',
                edgecolor='#0284c7',
                linewidth=1.2,
                alpha=0.35
            )
            self.ax_overlap.add_patch(poly_left)
            self.overlap_patches.append(poly_left)

        for z_depth, face, projected_2d in right_faces:
            poly_right = Polygon(
                projected_2d,
                closed=True,
                facecolor='#f43f5e',
                edgecolor='#e11d48',
                linewidth=1.2,
                alpha=0.35
            )
            self.ax_overlap.add_patch(poly_right)
            self.overlap_patches.append(poly_right)

    def _rasterize_faces(self, camera: PinholeCamera, visible_faces: List[Tuple[float, Face3D, np.ndarray]]) -> np.ndarray:
        """
        Rasterize projected 3D faces into a 2D BGR numpy image array matching camera pixel resolution.
        Includes high-contrast dot grid rendering if enable_dots is True.
        """
        w, h = camera.width, camera.height
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (42, 23, 15)  # Dark BGR background (15, 23, 42)

        for z_depth, face, projected_2d in visible_faces:
            shaded_color = face.compute_shaded_color(self.light_dir)
            r = int(shaded_color[0] * 255)
            g = int(shaded_color[1] * 255)
            b = int(shaded_color[2] * 255)
            pts = np.int32([projected_2d])
            cv2.fillPoly(img, pts, (b, g, r))
            cv2.polylines(img, pts, isClosed=True, color=(0, 0, 0), thickness=1)

            # Draw surface dot grid textures into image buffer for stereo matching
            if self.enable_dots:
                dots_3d = face.generate_surface_dots(grid_res=6)
                if len(dots_3d) > 0:
                    dots_2d, z_depths = camera.project_vertices(dots_3d, return_depth=True)
                    valid_dots = (z_depths > 0.1) & \
                                 (dots_2d[:, 0] >= 0) & (dots_2d[:, 0] < w) & \
                                 (dots_2d[:, 1] >= 0) & (dots_2d[:, 1] < h)
                    for px, py in np.int32(dots_2d[valid_dots]):
                        cv2.circle(img, (px, py), radius=3, color=(255, 255, 255), thickness=-1)
                        cv2.circle(img, (px, py), radius=4, color=(0, 0, 0), thickness=1)

        return img

    def update_depth_map(self, left_faces: List[Tuple[float, Face3D, np.ndarray]], right_faces: List[Tuple[float, Face3D, np.ndarray]]) -> None:
        """
        Extract depth information directly from the scene views using the stereo depth extraction pipeline
        and render it on the bottom-right subplot.
        """
        # Clear previous frame's dot artists in matplotlib
        for artist in self.dot_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.dot_artists.clear()

        # Rasterize left and right views into images
        img_left = self._rasterize_faces(self.stereo_rig.left_camera, left_faces)
        img_right = self._rasterize_faces(self.stereo_rig.right_camera, right_faces)

        # Run epipolar sliding window disparity calculation
        disp_map, valid_mask = self.matcher.compute_disparity(img_left, img_right)

        # Update depth calculator intrinsics/extrinsics
        self.depth_calculator.focal_length = self.stereo_rig.focal_length
        self.depth_calculator.baseline = self.stereo_rig.baseline
        self.depth_calculator.cx = self.stereo_rig.left_camera.cx
        self.depth_calculator.cy = self.stereo_rig.left_camera.cy

        # Convert disparity map to metric depth map Z = (f * B) / (d + doffs)
        depth_map = self.depth_calculator.disparity_to_depth(disp_map, min_depth=0.1, max_depth=15.0)

        # Mask out invalid occluded pixels for clean display
        display_depth = depth_map.copy()
        display_depth[~valid_mask] = 0.0

        if self.depth_im is None:
            self.ax_depth.clear()
            self.depth_im = self.ax_depth.imshow(display_depth, cmap='plasma', vmin=0.0, vmax=12.0)
            self.ax_depth.set_xlim(0, self.stereo_rig.width)
            self.ax_depth.set_ylim(self.stereo_rig.height, 0)
            self.ax_depth.set_aspect('equal')
            mode_title = "Extracted 3D Depth Map (Textured Surface Dot Grid Pipeline)" if self.enable_dots else "Extracted 3D Depth Map (Direct Epipolar Pipeline)"
            self.ax_depth.set_title(mode_title, fontsize=10.5, fontweight='bold', color='#10b981', pad=6)
            self.depth_cbar = self.fig.colorbar(self.depth_im, ax=self.ax_depth, fraction=0.046, pad=0.04)
            self.depth_cbar.ax.tick_params(colors='#94a3b8', labelsize=8)
            self.depth_cbar.set_label('Depth (m)', color='#10b981', fontsize=9)
        else:
            self.depth_im.set_data(display_depth)
            self.depth_im.set_clim(vmin=0.0, vmax=max(5.0, float(np.max(display_depth))))

    def update(self) -> None:
        """Render Left, Right, Overlapped, and Extracted Depth views simultaneously."""
        left_faces = self._render_camera_view(self.stereo_rig.left_camera, self.ax_left, self.left_patches)
        right_faces = self._render_camera_view(self.stereo_rig.right_camera, self.ax_right, self.right_patches)

        # Render real-time overlapped perspective in bottom-left
        self._render_overlapped_view(left_faces, right_faces)

        # Extract & render depth map on bottom-right subplot
        self.update_depth_map(left_faces, right_faces)

        fov_deg = self.stereo_rig.get_fov_degrees()
        pos_str = f"X: {self.stereo_rig.pos_x:+.2f}, Y: {self.stereo_rig.pos_y:+.2f}, Z: {self.stereo_rig.pos_z:+.2f}"
        texture_str = "ON (High-Contrast Dot Grid)" if self.enable_dots else "OFF (Solid Opaque Surfaces)"
        self.info_text.set_text(
            f"Stereo Rig Center: {pos_str}\n"
            f"Baseline B: {self.stereo_rig.baseline:.2f} m | FOV: {fov_deg:.1f}° | Surface Texture: {texture_str}\n"
            f"Left Cam X: {self.stereo_rig.left_camera.pos_x:+.2f} | Right Cam X: {self.stereo_rig.right_camera.pos_x:+.2f}\n"
            f"Faces Rendered: {len(left_faces)} (L) / {len(right_faces)} (R)"
        )

        self.fig.canvas.draw_idle()

    def add_sliders(self, focal_range=(200, 2000, 10), baseline_range=(0.05, 1.0, 0.01)) -> None:
        """Add focal length and baseline distance interactive sliders."""
        ax_focal = plt.axes([0.15, 0.025, 0.32, 0.025], facecolor='#1e293b')
        ax_baseline = plt.axes([0.58, 0.025, 0.32, 0.025], facecolor='#1e293b')

        self.slider_focal = Slider(
            ax_focal, 'Focal Length (f)', focal_range[0], focal_range[1],
            valinit=self.stereo_rig.focal_length, valstep=focal_range[2], color='#38bdf8'
        )
        self.slider_baseline = Slider(
            ax_baseline, 'Baseline (B)', baseline_range[0], baseline_range[1],
            valinit=self.stereo_rig.baseline, valstep=baseline_range[2], color='#a855f7'
        )

        def on_focal_change(val: float) -> None:
            self.stereo_rig.focal_length = val
            self.update()

        def on_baseline_change(val: float) -> None:
            self.stereo_rig.baseline = val
            self.update()

        self.slider_focal.on_changed(on_focal_change)
        self.slider_baseline.on_changed(on_baseline_change)

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
            elif key == 'p':
                self.export_stereo_pair()
                return True
            return False

        self.nav_controller.bind_controls(
            camera=self.stereo_rig,
            initial_pos=self.initial_pos,
            step_xy=step_xy,
            step_z=step_z,
            allow_3d=True,
            on_custom_key=handle_room_keys
        )

    def export_stereo_pair(self, output_dir: str = "data/synthetic_room") -> Tuple[str, str, str]:
        """
        Export Left view (im0.png), Right view (im1.png), and calibration file (calib.txt)
        formatted for robotics_neo depth extraction pipeline.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        im0_path = os.path.join(output_dir, "im0.png")
        im1_path = os.path.join(output_dir, "im1.png")
        calib_path = os.path.join(output_dir, "calib.txt")

        extent_left = self.ax_left.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(im0_path, bbox_inches=extent_left, dpi=100)

        extent_right = self.ax_right.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(im1_path, bbox_inches=extent_right, dpi=100)

        f = self.stereo_rig.focal_length
        cx = self.stereo_rig.left_camera.cx
        cy = self.stereo_rig.left_camera.cy
        b = self.stereo_rig.baseline
        w = self.stereo_rig.width
        h = self.stereo_rig.height

        calib_content = (
            f"cam0=[{f:.2f} 0 {cx:.2f}; 0 {f:.2f} {cy:.2f}; 0 0 1]\n"
            f"cam1=[{f:.2f} 0 {cx:.2f}; 0 {f:.2f} {cy:.2f}; 0 0 1]\n"
            f"doffs=0.0\n"
            f"baseline={b:.4f}\n"
            f"width={w}\n"
            f"height={h}\n"
            f"ndisp=64\n"
            f"isint=0\n"
        )

        with open(calib_path, "w") as f_out:
            f_out.write(calib_content)

        print(f"Exported stereo pair to {output_dir}: im0.png, im1.png, calib.txt")
        return im0_path, im1_path, calib_path

    def jump_to_waypoint(self, index: int) -> None:
        """Move stereo rig directly to a trajectory waypoint."""
        if 0 <= index < len(self.WAYPOINTS):
            name, (x, y, z) = self.WAYPOINTS[index]
            self.stereo_rig.pos_x, self.stereo_rig.pos_y, self.stereo_rig.pos_z = x, y, z
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
                self.stereo_rig.pos_x, self.stereo_rig.pos_y, self.stereo_rig.pos_z = float(pt[0]), float(pt[1]), float(pt[2])
                self.update()

            self.anim = animation.FuncAnimation(self.fig, animate_frame, frames=len(path_points), interval=50, blit=False)
            self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Show interactive Matplotlib window."""
        self.update()
        plt.show()
