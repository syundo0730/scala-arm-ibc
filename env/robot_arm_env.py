from math import pi, radians
from typing import Optional, Tuple, Set

import trio
from gym import spaces
import numpy as np

from camera.image import ImageShape
from env.robot_infra import RobotArmInfra
from env.async_step_env import AsyncStepEnv

_TOW_PI = 2 * pi


class RobotArmEnv(AsyncStepEnv):
    _DEFAULT_RESET_POSITION = np.array((0.2, 0))

    def __init__(self, delta_time: float,
                 observations: Optional[Set],
                 image_shape: Optional[ImageShape],
                 reset_position: Optional[np.ndarray] = None):
        super().__init__(delta_time)
        self.observation_space = self._create_observation_space(observations, image_shape)
        self._reset_position = reset_position if reset_position is not None else self._DEFAULT_RESET_POSITION
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))  # dx, dy

    @staticmethod
    def _create_observation_space(observations: Set, image_shape: Optional[ImageShape]) -> spaces.Dict:
        obs_scape_dict = {}
        if observations is None or 'joint_angles' in observations:
            obs_scape_dict['joint_angles'] = spaces.Box(
                low=-_TOW_PI, high=_TOW_PI, shape=(2,)
            )
        if observations is None or 'end_effector_pos' in observations:
            obs_scape_dict['end_effector_pos'] = spaces.Box(
                low=-.3, high=.3, shape=(2,)
            )
        if (observations is None or 'rgb' in observations) and image_shape is not None:
            obs_scape_dict['rgb'] = spaces.Box(
                low=0, high=255,
                shape=image_shape.np_array_shape,
                dtype=np.uint8)
        return spaces.Dict(obs_scape_dict)

    async def async_reset(self, infra: RobotArmInfra) -> None:
        await infra.move_end_effector_to(self._reset_position, joint_speed=radians(60))
        await trio.sleep(3)
        await super().async_reset(infra)
