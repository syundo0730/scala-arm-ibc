from functools import partial
from typing import Optional, Tuple, Sequence

import trio
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from gym import register

from camera.image import ImageShape
from controller.policy_executor import PolicyExecutor
from env.robot_arm_env import RobotArmEnv


@gin.configurable
async def policy_eval(
        serial_port_name: str,
        saved_model_path: str,
        checkpoint_path: str,
        env_name='',
        sequence_length=2,
        target_update_delta_time=0.1,
        command_delta_time=0.01,
        observations: Optional[Sequence] = None,
        image_shape: Optional[Tuple[float, float]] = None,  # w, h, channel
):
    if image_shape:
        image_shape = ImageShape(*image_shape)

    register(
        id='ScalaArm-v0',
        entry_point=RobotArmEnv,
    )
    executor = PolicyExecutor(saved_model_path, checkpoint_path)
    await executor.run(
        env_name,
        sequence_length,
        target_update_delta_time,
        command_delta_time,
        observations,
        image_shape,
        serial_port_name
    )


FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_string(
    'serial_port', '',
    'name of serial port e.g.: /dev/tty.usbserial-001')
flags.DEFINE_string(
    'saved_model_path', '',
    'path of saved policy')
flags.DEFINE_string(
    'checkpoint_path', '',
    'path of checkpoint of policy')


def main(_):
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings, skip_unknown=False)

    tf.config.run_functions_eagerly(False)

    trio.run(partial(
        policy_eval,
        serial_port_name=FLAGS.serial_port,
        saved_model_path=FLAGS.saved_model_path,
        checkpoint_path=FLAGS.checkpoint_path,
    ))


if __name__ == '__main__':
    app.run(main)
