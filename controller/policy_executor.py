from typing import Optional, Sequence

import trio
from tf_agents.environments import suite_gym, HistoryWrapper
from tf_agents.policies import SavedModelPyTFEagerPolicy

from camera.image import ImageShape
from env.robot_arm_real_infra import open_arm_control


class PolicyExecutor:
    def __init__(self, saved_model_path: str, checkpoint_path: str):
        policy = SavedModelPyTFEagerPolicy(
            saved_model_path, load_specs_from_pbtxt=True)
        policy.update_from_checkpoint(checkpoint_path)
        self._policy = policy

    async def run(
            self,
            env_name: str,
            sequence_length: int,
            target_update_delta_time: float,
            command_delta_time: float,
            observations: Optional[Sequence] = None,
            image_shape: Optional[ImageShape] = None,
            serial_port_name: str = '',
    ):
        env = suite_gym.load(env_name, gym_kwargs={
            'delta_time': target_update_delta_time,
            'observations': observations,
            'image_shape': image_shape,
        })
        env = HistoryWrapper(env, history_length=sequence_length, tile_first_step_obs=True)

        async with open_arm_control(
                serial_port_name, target_update_delta_time, command_delta_time, observations) as robot_infra:
            await env.async_reset(robot_infra)
            time_step = env.reset()
            while True:
                start_at = trio.current_time()
                action_step = self._policy.action(time_step)
                # print('estimate end', trio.current_time() - start_at)
                action = action_step.action
                # print(action)
                await env.async_step(robot_infra, action)
                time_step = env.step(None)
                # print('one cycle end: ', trio.current_time() - start_at)
