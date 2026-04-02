"""Dynamic scene rendering with moving items and fixed camera."""

import logging
import os
import os.path as osp
from abc import ABC, abstractmethod

import mitsuba as mi
from tqdm import tqdm

logger = logging.getLogger(__name__)

from projects.base import BaseProject
from scene_builder.mitsuba_utils import set_camera_pose
from utils.registry import PROJECT_REGISTRY
from utils.video_utils import gen_video_from_folder


@PROJECT_REGISTRY.register()
class MovingItem(BaseProject, ABC):
    """
    Base class for rendering dynamic scenes with moving items.

    The camera remains fixed while specific scene items move between frames.
    Subclasses must implement update_move() to define the motion.

    Useful for creating demonstration videos showing material appearance under
    varying conditions (e.g., moving lights for SVBRDF visualization).

    Config options:
        img_num (int): Number of frames to render (default: 10)

    Abstract methods:
        update_move(): Update scene item positions for the next frame
    """

    def setup_output_paths(self):
        """Configure output paths for image sequence."""
        super().setup_output_paths()
        self.img_num = self.conf.get('img_num', 10)
        self.export_video = self.conf.get('export_video', False)
        self.video_fps = self.conf.get('video_fps', 30)
        self.image_folder = osp.join(self.output_folder, 'image')
        os.makedirs(self.image_folder, exist_ok=True)

    @abstractmethod
    def update_move(self):
        """
        Update scene item positions for the next frame.

        This method is called before each frame is rendered and should modify
        scene parameters (e.g., vertex positions, transforms) to create motion.
        Must be implemented by subclasses.
        """
        pass

    def run(self):
        """Render animation sequence with fixed camera and moving items."""
        # Set fixed camera pose
        set_camera_pose(self.camera, **self.pose_generator.pose_list[0])

        # Render each frame
        for idx in tqdm(range(self.img_num), desc="Rendering frames"):
            # Update item positions
            self.update_move()

            # Render frame
            image = mi.render(self.scene.scene, sensor=self.camera)

            # Save frame
            img_path = self.get_path(idx)
            mi.util.write_bitmap(img_path, image)

        if self.export_video:
            video_path = osp.join(self.output_folder, 'video.mp4')
            gen_video_from_folder(self.image_folder, video_path, fps=self.video_fps)
            logger.info(f"Video saved to: {video_path}")
