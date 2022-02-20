from abc import abstractmethod
import numpy as np
from typing import Dict, Optional


class RobotInfra:
    @abstractmethod
    def action_no_wait(self, action) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_observation(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_observation_no_wait(self) -> Dict:
        raise NotImplementedError


class RobotArmInfra(RobotInfra):
    @abstractmethod
    def action_now_wait(self, xy_diff: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_observation(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_observation_no_wait(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def action_absolute_no_wait(self, xy: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    async def run(self):
        raise NotImplementedError
