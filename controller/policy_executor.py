import trio
from tf_agents.environments import suite_gym, HistoryWrapper
from tf_agents.policies import SavedModelPyTFEagerPolicy

from controller.oracle_recorder import OracleRecorder, IMAGE_SHAPE
from env.robot_arm_real_infra import open_arm_control


class PolicyExecutor:
    _HISTORY_LENGTH = 2

    def __init__(self, saved_model_path: str, checkpoint_path: str):
        policy = SavedModelPyTFEagerPolicy(
            saved_model_path, load_specs_from_pbtxt=True)
        policy.update_from_checkpoint(checkpoint_path)
        self._policy = policy

    async def run(self):
        env = suite_gym.load('ScalaArm-v0', gym_kwargs={
            'delta_time': OracleRecorder.DELTA_TIME,
            'image_shape': IMAGE_SHAPE,
        })
        env = HistoryWrapper(
            env, history_length=self._HISTORY_LENGTH, tile_first_step_obs=True)

        async with open_arm_control(
                '/dev/ttyUSB0', observations=['end_effector_pos'],
                target_update_delta_time=OracleRecorder.DELTA_TIME,
                command_delta_time=OracleRecorder.COMMAND_DELTA_TIME) as robot_infra:
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
