"""Planar light calibration data generation project."""

import os
import os.path as osp

import mitsuba as mi
import numpy as np

from projects.base import BaseProject
from scene_builder.mitsuba_utils import set_camera_pose
from utils.registry import PROJECT_REGISTRY


@PROJECT_REGISTRY.register()
class PlanarCalibration(BaseProject):
    """
    Generates calibration test data for SVBRDF capture under planar lighting.

    Renders images with a fixed camera and a moving checkerboard pattern,
    recording checkerboard poses for verification. Follows the calibration
    workflow described in Zhang et al. 2023 for planar light sources.

    Outputs:
        - target_calibration.png: Original checkerboard position
        - helper/helper_*.png: 10 random checkerboard positions
        - light.png: Checkerboard on planar light surface
        - params.txt: All checkerboard and camera poses

    The scene must include a 'checkerboard' element for this project to work.
    """

    def __init__(self, conf):
        super().__init__(conf)

        # Record the original checkerboard transform (mainly for its scale)
        params = mi.traverse(self.scene.scene)
        self.basic_transform = np.array(params['checkerboard.to_world'].matrix)
        self.basic_transform = mi.Transform4f(self.basic_transform)

        # Need camera params to place checkerboard within the image
        self.camera_origin = self.pose_generator.pose['origin']
        self.camera_target = self.pose_generator.pose['target']

        # String template for recording poses
        self.str_template = '{} pose: origin: [{:.4f},{:.4f},{:.4f}], target: [{:.4f},{:.4f},{:.4f}]\n'

    def setup_output_paths(self):
        """Configure output paths for calibration images and parameters."""
        super().setup_output_paths()
        self.target_img_path = osp.join(self.output_folder, 'target_calibration.png')
        self.helper_folder = osp.join(self.output_folder, 'helper')
        os.makedirs(self.helper_folder, exist_ok=True)
        self.helper_img_path_list = [
            osp.join(self.helper_folder, 'helper_{:02d}.png'.format(i))
            for i in range(10)
        ]
        self.light_img_path = osp.join(self.output_folder, 'light.png')
        self.params_path = osp.join(self.output_folder, 'params.txt')
        self.params_str = ''

    def run(self):
        """Generate calibration images with checkerboard at various poses."""
        camera_pose = self.pose_generator.pose
        set_camera_pose(self.camera, **camera_pose)

        # Record camera pose
        self.params_str += self.str_template.format(
            'camera', *camera_pose['origin'], *camera_pose['target']
        )

        # Record target checkerboard location
        self.params_str += self.str_template.format(
            'target checkerboard', *[0, 0, 0], *[0, 0, 1]
        )

        # Render the original checkerboard as the sample target
        image = mi.render(self.scene.scene, sensor=self.camera)
        mi.util.write_bitmap(self.target_img_path, image)

        # Randomly move the checkerboard and render helper images
        for idx in range(len(self.helper_img_path_list)):
            self.update_scene_for_calibration(idx)
            image = mi.render(self.scene.scene, sensor=self.camera)
            mi.util.write_bitmap(self.helper_img_path_list[idx], image)

        # Render the checkerboard on the planar light
        self.update_scene_for_light()
        image = mi.render(self.scene.scene, sensor=self.camera)
        mi.util.write_bitmap(self.light_img_path, image)

        # Save all recorded poses
        with open(self.params_path, 'w') as file:
            print(self.params_str, file=file)

    def update_scene_for_calibration(self, idx):
        """Randomly position checkerboard for calibration helper image."""
        # Random position and orientation
        origin = np.random.uniform(-0.5, 0.5, 3) + self.camera_target
        target = np.random.uniform(-0.5, 0.5, 3) + self.camera_origin
        transform = mi.Transform4f.look_at(origin=origin, target=target, up=[0, 0, 1])

        # Update checkerboard transform
        params = mi.traverse(self.scene.scene)
        params['checkerboard.to_world'] = transform @ self.basic_transform
        params.update()

        # Record pose
        self.params_str += self.str_template.format(
            'helper_{} checkerboard'.format(idx), *origin, *target
        )

    def update_scene_for_light(self):
        """Position checkerboard on planar light surface."""
        # Translate checkerboard to light plane
        transform = mi.Transform4f.translate([0, 0, -0.3])

        # Update checkerboard transform
        params = mi.traverse(self.scene.scene)
        params['checkerboard.to_world'] = transform @ self.basic_transform
        params.update()

        # Record pose
        self.params_str += self.str_template.format(
            'light checkerboard', *[0, 0, -0.3], *[0, 0, 1]
        )
