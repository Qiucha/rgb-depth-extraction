import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Callable, Optional
from .geometry import Cube3D
from .camera import PinholeCamera
from .controls import NavigationController


class CubeVisualizer:
    """Handles rendering and interaction of 3D cube projection using Matplotlib."""

    def __init__(self, camera: PinholeCamera, cube: Cube3D, title: str = "3D Cube Projection"):
        self.camera = camera
        self.cube = cube
        self.title = title

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2)
        
        self.ax.set_xlim(0, self.camera.width)
        self.ax.set_ylim(self.camera.height, 0)  # Invert Y-axis for screen coordinates
        self.ax.set_title(self.title)

        # Lines for wireframe rendering
        self.lines = []
        for start, end in self.cube.edges:
            line, = self.ax.plot([], [], 'k-', lw=2)
            self.lines.append((line, start, end))

        self.initial_pos = (self.camera.pos_x, self.camera.pos_y, self.camera.pos_z)

        # Status text overlay (FOV & position)
        self.info_text = self.ax.text(
            0.02, 0.95, '', transform=self.ax.transAxes,
            fontsize=11, fontweight='bold', color='blue', va='top'
        )

        # Controls legend overlay
        self.controls_text = self.ax.text(
            0.98, 0.95, '', transform=self.ax.transAxes,
            fontsize=9, color='#333333', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9)
        )

        self.slider = None

    def update(self):
        """Re-project vertices and redraw line coordinates."""
        projected_2d = self.camera.project_vertices(self.cube.vertices)

        for line, start, end in self.lines:
            x_data = [projected_2d[start][0], projected_2d[end][0]]
            y_data = [projected_2d[start][1], projected_2d[end][1]]
            line.set_data(x_data, y_data)

        # Update FOV & Position text
        fov_deg = self.camera.get_fov_degrees()
        pos_str = f"({self.camera.pos_x:.1f}, {self.camera.pos_y:.1f}, {self.camera.pos_z:.1f})"
        self.info_text.set_text(f"FOV: {fov_deg:.1f}°\nCamera Pos: {pos_str}")

        self.fig.canvas.draw_idle()

    def add_focal_length_slider(self, min_val: float = 100, max_val: float = 2000, step: float = 10):
        """Add interactive focal length slider."""
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Focal Length', min_val, max_val, valinit=self.camera.focal_length, valstep=step)

        def on_slider_change(val):
            self.camera.focal_length = val
            self.update()

        self.slider.on_changed(on_slider_change)

    def bind_keyboard_navigation(self, step_xy: float = 0.3, step_z: float = 0.2, allow_3d: bool = True):
        """Bind keyboard controls using NavigationController."""
        legend = NavigationController.get_navigation_legend(allow_3d=allow_3d, include_waypoints=False)
        self.controls_text.set_text(legend)

        self.nav_controller = NavigationController(self.fig.canvas, self.update)
        self.nav_controller.bind_controls(
            camera=self.camera,
            initial_pos=self.initial_pos,
            step_xy=step_xy,
            step_z=step_z,
            allow_3d=allow_3d
        )

    def show(self):
        """Perform initial update and show the plot window."""
        self.update()
        plt.show()
