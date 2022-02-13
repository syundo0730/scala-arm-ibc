from math import pi, radians
from typing import Optional, Tuple

import trio
from gym import spaces
import numpy as np

from env.robot_infra import RobotArmInfra
from env.async_step_env import AsyncStepEnv

_TOW_PI = 2 * pi


class RobotArmEnv(AsyncStepEnv):
    _DEFAULT_RESET_POSITION = np.array((0.2, 0))

    def __init__(self, infra: RobotArmInfra, delta_time: float, image_size: Optional[Tuple[float, float]],
                 reset_position: Optional[np.ndarray] = None):
        super().__init__(infra, delta_time)
        self._infra = infra
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))  # dx, dy
        self.observation_space = self._create_observation_space(image_size)
        self._reset_position = reset_position if reset_position is not None else self._DEFAULT_RESET_POSITION

    @staticmethod
    def _create_observation_space(image_size: Optional[Tuple[float, float]]) -> spaces.Dict:
        obs_scape_dict = {
            'joint_angles': spaces.Box(
                low=-_TOW_PI, high=_TOW_PI, shape=(2,)
            ),
            'end_effector_pos': spaces.Box(
                low=-.3, high=.3, shape=(2,)
            ),
        }
        if image_size is not None:
            obs_scape_dict['rgb'] = spaces.Box(
                low=0, high=255,
                shape=(*image_size, 3),
                dtype=np.uint8)
        return spaces.Dict(obs_scape_dict)

    async def async_reset(self) -> None:
        await self._infra.reset()
        await self._infra.move_end_effector_to(self._reset_position, joint_speed=radians(60))
        await trio.sleep(3)
        await super().async_reset()
