"""Moving point light source project for SVBRDF demonstration."""

import numpy as np
import mitsuba as mi

from projects.MovingItem import MovingItem
from utils.registry import PROJECT_REGISTRY


@PROJECT_REGISTRY.register()
class MovingPointLight(MovingItem):
    """
    Demonstrates SVBRDF appearance under a circularly orbiting point light.

    The light orbits around a center point in a circular path on the XY plane
    (at the initial Z height).

    Config options:
        light_center (list): Orbit center [x, y, z] (default: [0, 0, 0])
        light_radius (float): Orbit radius (default: auto from initial position)
        img_num (int): Number of frames (inherited from MovingItem)
    """

    def __init__(self, conf):
        self.light_center = np.array(conf.get('light_center', [0, 0, 0]), dtype=float)
        self._configured_radius = conf.get('light_radius', None)
        self._frame_idx = 0
        super().__init__(conf)

        # Compute radius and initial angle from the light's starting position
        params = mi.traverse(self.scene.scene)
        pos = params['point_light.position']
        init_pos = np.array([pos.x[0], pos.y[0], pos.z[0]])
        self._height = init_pos[2]
        offset = init_pos[:2] - self.light_center[:2]
        self._radius = (
            float(self._configured_radius)
            if self._configured_radius is not None
            else float(np.linalg.norm(offset))
        )
        self._start_angle = float(np.arctan2(offset[1], offset[0]))

    def update_move(self):
        """Orbit the point_light around center."""
        angle = self._start_angle + 2 * np.pi * self._frame_idx / self.img_num
        new_pos = np.array([
            self.light_center[0] + self._radius * np.cos(angle),
            self.light_center[1] + self._radius * np.sin(angle),
            self._height,
        ], dtype=np.float32)
        params = mi.traverse(self.scene.scene)
        params['point_light.position'] = new_pos
        params.update()
        self._frame_idx += 1
