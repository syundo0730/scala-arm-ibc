from abc import abstractmethod
import numpy as np
from typing import Dict, Optional


class RobotInfra:
    @abstractmethod
    async def command_action(self, action) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_observation(self) -> Dict:
        raise NotImplementedError


class RobotArmInfra(RobotInfra):
    @abstractmethod
    async def command_action(self, xy_diff: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_observation(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    async def move_end_effector_to(self, xy: np.ndarray, joint_speed: Optional[float] = None):
        raise NotImplementedError

    @abstractmethod
    async def reset(self):
        pass
