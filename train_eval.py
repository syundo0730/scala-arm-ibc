import datetime
import os
from functools import partial

import trio
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from gym import register
from tf_agents.environments import PyEnvironment, suite_gym, HistoryWrapper
from tf_agents.train.utils.spec_utils import get_tensor_specs
from tf_agents.train.utils.strategy_utils import get_strategy
from tf_agents.train.utils.train_utils import create_train_step
from trio_serial import SerialStream

from camera.video_capture import open_video_capture
from controller.oracle_recorder import OracleRecorder, IMAGE_SIZE
from env.robot_arm_env import RobotArmEnv
from env.robot_arm_real_infra import RobotArmRealInfra, open_arm_control
from ibc.ibc.train.get_agent import get_agent
from ibc.ibc.train.get_cloning_network import get_cloning_network
from ibc.ibc.train.get_data import get_data_fns
from ibc.ibc.train.get_learner import get_learner
from ibc.ibc.train.get_normalizers import get_normalizers
from ibc.ibc.train.get_sampling_spec import get_sampling_spec
from ibc.ibc.train_eval import get_distributed_eval_data, validation_step, training_step


@gin.configurable
def train_eval_simple(
        # basic configs
        env: PyEnvironment,
        strategy,
        sequence_length=2,
        root_dir='',
        dataset_path='',
        network_name='MLPEBM',  # 'MLPEBM', 'MLPMSE', 'MLPMDN', 'ConvMLPMSE', 'ConvMLPMDN', 'PixelEBM'
        loss_type_name='ebm',  # 'ebm' or 'mse' or 'mdn'.
        eval_loss_interval=100,
        seed=0,
        add_time_to_log=True,
        # training params
        batch_size=512,
        dataset_eval_fraction=0.0,
        decay_steps=100,
        fused_train_steps=100,
        learning_rate=1e-3,
        max_data_shards=-1,  # -1 for 'use all'.
        num_iterations=20000,
        replay_capacity=100000,
        uniform_boundary_buffer=0.05,
        use_warmup=False
):
    tf.random.set_seed(seed)
    if not tf.io.gfile.exists(root_dir):
        tf.io.gfile.makedirs(root_dir)
    if add_time_to_log:
        root_dir = os.path.join(root_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = get_tensor_specs(env)
    for_rnn = False
    flatten_action = True
    create_train_and_eval_fns_un_normalized = get_data_fns(
        dataset_path,
        sequence_length,
        replay_capacity,
        batch_size,
        for_rnn,
        dataset_eval_fraction,
        flatten_action)
    train_data, _ = create_train_and_eval_fns_un_normalized()
    norm_info, norm_train_data_fn = get_normalizers(
        train_data, batch_size, env_name='')  # give arbitrary dummy name, because it's not important.

    per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
    create_train_and_eval_fns = get_data_fns(
        dataset_path,
        sequence_length,
        replay_capacity,
        per_replica_batch_size,
        for_rnn,
        dataset_eval_fraction,
        flatten_action,
        norm_function=norm_train_data_fn,
        max_data_shards=max_data_shards)
    validation_data_iter = get_distributed_eval_data(create_train_and_eval_fns, strategy)

    with strategy.scope():
        # Create train step counter.
        train_step = create_train_step()

        # Define action sampling spec.
        action_sampling_spec = get_sampling_spec(
            action_tensor_spec,
            min_actions=norm_info.min_actions,
            max_actions=norm_info.max_actions,
            uniform_boundary_buffer=uniform_boundary_buffer,
            act_norm_layer=norm_info.act_norm_layer)

        # This is a common opportunity for a bug, having the wrong sampling min/max
        # so log this.
        logging.info(('Using action_sampling_spec:', action_sampling_spec))

        cloning_network = get_cloning_network(
            network_name,
            obs_tensor_spec,
            action_tensor_spec,
            norm_info.obs_norm_layer,
            norm_info.act_norm_layer,
            sequence_length,
            norm_info.act_denorm_layer)
        agent = get_agent(
            loss_type_name,
            time_step_tensor_spec,
            action_tensor_spec,
            action_sampling_spec,
            norm_info.obs_norm_layer,
            norm_info.act_norm_layer,
            norm_info.act_denorm_layer,
            learning_rate,
            use_warmup,
            cloning_network,
            train_step,
            decay_steps)
        bc_learner = get_learner(
            loss_type_name,
            root_dir,
            agent,
            train_step,
            create_train_and_eval_fns,
            fused_train_steps,
            strategy)
        get_eval_loss = tf.function(agent.get_eval_loss)

    logging.info('Saving operative-gin-configs.')
    with tf.io.gfile.GFile(
            os.path.join(root_dir, 'operative-gin-configs.txt'), 'wb') as f:
        f.write(gin.operative_config_str())

    # Main train and eval loop.
    while train_step.numpy() < num_iterations:
        # Run bc_learner for fused_train_steps.
        training_step(agent, bc_learner, fused_train_steps, train_step)

        if validation_data_iter is not None and train_step.numpy() % eval_loss_interval == 0:
            # Run a validation step.
            validation_step(
                validation_data_iter, bc_learner, train_step, get_eval_loss)


@gin.configurable
async def train_eval_with_real_robot(
        serial_port_name: str,
        strategy,
        env_name='',
        observations=None,
        image_size=None,
        sequence_length=2,
        **kwargs
):
    register(
        id='ScalaArm-v0',
        entry_point=RobotArmEnv,
    )
    env = suite_gym.load(env_name, gym_kwargs={
        'delta_time': OracleRecorder.DELTA_TIME,
        'observations': observations,
        'image_size': IMAGE_SIZE,
    })
    env = HistoryWrapper(env, history_length=sequence_length, tile_first_step_obs=True)
    train_eval_simple(env, strategy, sequence_length=sequence_length, **kwargs)
    async with open_arm_control(serial_port_name, observations, image_size) as robot_infra:
        pass


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'serial_port', '',
    'name of serial port e.g.: /dev/tty.usbserial-001')


def main(_):
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings, skip_unknown=False)

    tf.config.run_functions_eagerly(False)

    strategy = get_strategy(tpu=FLAGS.tpu, use_gpu=FLAGS.use_gpu)
    trio.run(partial(
        train_eval_with_real_robot,
        serial_port_name=FLAGS.serial_port,
        strategy=strategy,
        add_time_to_log=FLAGS.add_time,
    ))


if __name__ == '__main__':
    app.run(main)
