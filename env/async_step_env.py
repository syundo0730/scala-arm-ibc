from typing import Tuple

import gym
import trio

from env.robot_infra import RobotInfra


class AsyncStepEnv(gym.Env):
    def __init__(self, delta_time: float):
        self._delta_time = delta_time

        self._done = False
        self._last_observation = None
        self._last_observed_time = None
        self._last_reward = 0
        self._is_observation_fresh = False

    def step(self, action):
        assert self._is_observation_fresh, 'async_step should be called before'
        self._is_observation_fresh = False
        return self._last_observation, self._last_reward, self._done, {}

    def reset(self):
        assert self._is_observation_fresh, 'async_reset should be called before'
        self._is_observation_fresh = False
        return self._last_observation

    def render(self, mode='human'):
        pass

    async def _sleep_for_observation(self):
        if self._last_observed_time is None:
            await trio.sleep(0)
        else:
            await trio.sleep_until(self._last_observed_time + self._delta_time)

    @staticmethod
    def _calc_reward_and_done(observation, action) -> Tuple[float, bool]:
        # reward, done
        return 0, False

    async def async_step(self, infra: RobotInfra, action) -> None:
        await infra.command_action(action)
        await self._sleep_for_observation()
        self._last_observation = await infra.get_observation()
        self._last_observed_time = trio.current_time()
        self._last_reward, self._done = self._calc_reward_and_done(self._last_observation, action)
        self._is_observation_fresh = True

    async def async_reset(self, infra: RobotInfra) -> None:
        self._last_observation = await infra.get_observation()
        self._last_observed_time = trio.current_time()
        self._is_observation_fresh = True
