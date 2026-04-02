"""Single-pose camera generators."""
import numpy as np

from utils.tool_utils import spherical2cartesian
from utils.registry import CAMPOSE_REGISTRY


@CAMPOSE_REGISTRY.register()
class SinglePose:
    """Single camera pose defined either explicitly or via spherical angles."""

    def __init__(self, conf):
        self.conf = conf
        self._pose = self._build_pose()
        self.pose_list = [self._pose]
        self.img_num = 1

    @property
    def pose(self):
        """Deprecated: use pose_list[0] instead. Kept for backward compatibility."""
        return self._pose

    def _build_pose(self):
        if 'theta' in self.conf:
            theta = self.conf.get('theta',45.0)
            phi = self.conf.get('phi',0.0)
            radius = self.conf.get('r',1.0)
            shift = np.array(self.conf.get('shift',[0.0, 0.0, 0.0]), dtype=np.float32)
            origin = spherical2cartesian(theta, phi, radius) + shift
        else:
            origin = np.array(self.conf.get('origin',[0.0, 0.0, 0.0]), dtype=np.float32)

        target = self.conf.get('target',[0.0, 0.0, 0.0])
        up = self.conf.get('up',[0.0, 0.0, 1.0])
        return {
            'origin': origin,
            'target': target,
            'up': up,
        }


@CAMPOSE_REGISTRY.register()
class RandomPose:
    def __init__(self, conf):
        self.conf = conf
        self.theta_range = conf.get('theta_range', default = [20,70])
        self.r_range = conf.get('r_range',default=[1.0,1.2])
        self.shift =np.array(self.conf.get('shift',[0,0,0]))
        self.target = conf.get('target', default = [0,0,0])

    def random_sample(self):
        # random sample a camera location within the range
        pose_list = []
        r = np.random.uniform(self.r_range[0], self.r_range[1])
        theta_range = [np.cos(np.deg2rad(self.theta_range[1])), np.cos(np.deg2rad(self.theta_range[0]))]
        v = np.random.uniform(*theta_range)
        theta = np.rad2deg(np.arccos(v))
        phi = np.random.uniform(0, 360)
        position = spherical2cartesian(theta, phi, r)
        position += self.shift
        dic = {}
        dic['origin'] = position
        dic['target'] = self.target
        dic['up'] = [0,0,1]

        return dic

    def random_multi_sample(self, sample_num):
        pose_list = []
        for _ in range(sample_num):
            dic = self.random_sample()
            pose_list.append(dic)
        return pose_list
