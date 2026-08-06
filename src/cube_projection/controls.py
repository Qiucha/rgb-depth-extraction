import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional


class NavigationController:
    """Encapsulates Matplotlib canvas keybinding cleanup, legend formatting, and keyboard movement events."""

    def __init__(self, canvas, on_update_callback: Callable[[], None]):
        """
        :param canvas: Matplotlib figure canvas.
        :param on_update_callback: Callback triggered when camera state changes to redraw scene.
        """
        self.canvas = canvas
        self.on_update_callback = on_update_callback
        self.cid: Optional[int] = None
        self.strip_default_keymaps()
        self.disconnect_existing_handlers()

    def strip_default_keymaps(self) -> None:
        """Disconnect Matplotlib default key press handler and strip conflicting shortcuts."""
        for k in ('save', 'quit', 'home'):
            key_list = plt.rcParams.get(f'keymap.{k}', [])
            for key_to_remove in ('s', 'q', 'r', '1', '2', '3', '4', '5'):
                if key_to_remove in key_list:
                    key_list.remove(key_to_remove)

    def disconnect_existing_handlers(self) -> None:
        """Disconnect existing canvas key press handlers if present."""
        if hasattr(self.canvas, 'manager') and self.canvas.manager:
            handler_id = getattr(self.canvas.manager, 'key_press_handler_id', None)
            if handler_id is not None:
                self.canvas.mpl_disconnect(handler_id)

    @staticmethod
    def get_navigation_legend(allow_3d: bool = True, include_waypoints: bool = False) -> str:
        """Generate controls legend text overlay."""
        legend = "Controls:\n"
        legend += "W / S or ↑ / ↓ : Camera Up / Down\n"
        legend += "A / D or ← / → : Camera Left / Right\n"
        if allow_3d:
            legend += "Q / E : Camera Back / Forward\n"
        if include_waypoints:
            legend += "1 - 5 : Trajectory Waypoints\n"
            legend += "T : Toggle Trajectory Tour\n"
        legend += "R : Reset Position"
        return legend

    def bind_controls(self, camera, initial_pos: Tuple[float, float, float],
                      step_xy: float = 0.3, step_z: float = 0.3, allow_3d: bool = True,
                      on_custom_key: Optional[Callable[[str], bool]] = None) -> None:
        """
        Bind keyboard events to camera motion.
        
        :param camera: PinholeCamera instance to mutate coordinates.
        :param initial_pos: (x, y, z) tuple to restore on reset key ('r').
        :param step_xy: Movement step along X and Y axes.
        :param step_z: Movement step along Z axis.
        :param allow_3d: Whether Z axis navigation is enabled.
        :param on_custom_key: Optional custom handler return True if key was handled.
        """
        def on_key_press(event) -> None:
            key = event.key.lower() if event.key else ''

            if on_custom_key and on_custom_key(key):
                self.on_update_callback()
                return

            if key in ('w', 'up'):
                camera.move(dy=-step_xy)
            elif key in ('s', 'down'):
                camera.move(dy=step_xy)
            elif key in ('a', 'left'):
                camera.move(dx=-step_xy)
            elif key in ('d', 'right'):
                camera.move(dx=step_xy)
            elif allow_3d and key == 'q':
                camera.move(dz=-step_z)
            elif allow_3d and key == 'e':
                camera.move(dz=step_z)
            elif key == 'r':
                camera.pos_x, camera.pos_y, camera.pos_z = initial_pos
            else:
                return

            self.on_update_callback()

        self.cid = self.canvas.mpl_connect('key_press_event', on_key_press)
