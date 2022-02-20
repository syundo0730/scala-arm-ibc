import math
from contextlib import nullcontext, asynccontextmanager
from math import degrees, radians
from typing import List, Dict, Optional, Sequence

import cv2
import numpy as np
import trio
from trio_serial import AbstractSerialStream, SerialStream
from trio_util import move_on_when, periodic, RepeatedEvent

from camera.image import ImageShape
from camera.video_capture import open_video_capture
from motors.core.serial_client import SerialClient
from env.arm_kinematics_2d import inverse, forward
from env.robot_infra import RobotArmInfra
from motors.gyems.rmd_r485.commands import Commands


class RobotArmRealInfra(RobotArmInfra):
    _MOTOR_IDS = (1, 2)
    _JOINT_SPEED = 360  # degree / sec
    _COLLIDABLE_DISTANCE_TO_BASE = 0.1

    def __init__(self, serial_stream: AbstractSerialStream, video_stream: cv2.VideoCapture,
                 target_update_delta_time: float,
                 command_delta_time: float,
                 observations: Optional[Sequence],
                 image_shape: Optional[ImageShape]):
        self._serial_client = SerialClient(serial_stream)
        self._video_stream = video_stream
        self._target_update_delta_time = target_update_delta_time
        self._command_delta_time = command_delta_time
        self._observations = observations
        self._image_shape = image_shape

        self._last_target_position = None
        self._target_update_event = RepeatedEvent()
        self._last_observation = {}
        self._observation_update_event = RepeatedEvent()

    def action_no_wait(self, xy_diff: np.ndarray) -> None:
        self._last_target_position += xy_diff
        self._target_update_event.set()

    def action_absolute_no_wait(self, xy: np.ndarray) -> None:
        self._last_target_position = xy
        self._target_update_event.set()

    def get_observation_no_wait(self) -> Dict:
        return self._last_observation

    async def get_observation(self) -> Dict:
        await self._observation_update_event.wait()
        return self._last_observation

    @property
    def target_position(self) -> Optional[np.ndarray]:
        return self._last_target_position

    async def _reset_pid_params(self):
        # write pid control params
        for id_ in self._MOTOR_IDS:
            await self._serial_client.command(Commands.write_control_param_ram(id_, 10, 100, 200, 30))
            print('set PID param of ', id_)

    async def _shutdown(self):
        for id_ in self._MOTOR_IDS:
            await self._serial_client.command(Commands.shutdown(id_))
            print(f'ID: {id_} shutdown')

    async def _move_end_effector_to(self, xy: np.ndarray, *, joint_speed: Optional[float] = None) -> None:
        dist_from_origin = math.hypot(*xy)
        if dist_from_origin < self._COLLIDABLE_DISTANCE_TO_BASE:
            await trio.sleep(0)
            return
        angle_0, angle_1 = inverse(xy)
        # convert radians to degrees, and invert the sign (from ccw positive to cw positive)
        speed = degrees(joint_speed) if joint_speed is not None else self._JOINT_SPEED
        commands = [Commands.move_to_position_with_speed(id_, -degrees(angle), speed)
                    for id_, angle in zip(self._MOTOR_IDS, (angle_0, angle_1))]
        for command in commands:
            await self._serial_client.command(command)

    async def _read_joint_angles(self) -> List:
        commands = [Commands.read_position(id_) for id_ in self._MOTOR_IDS]
        # convert from degrees to radians, and invert the sign (from cw positive to ccw positive)
        return [-radians((await self._serial_client.query(command)).angle) for command in commands]

    async def _fetch_observation(self) -> Dict:
        joint_angles = await self._read_joint_angles()
        end_effector_point = forward(*joint_angles)
        obs = {}
        if self._observations is None or 'joint_angles' in self._observations:
            obs['joint_angles'] = joint_angles
        if self._observations is None or 'end_effector_pos' in self._observations:
            obs['end_effector_pos'] = end_effector_point
        if (self._observations is None or 'rgb' in self._observations) and self._image_shape is not None:
            success, frame = self._video_stream.read()
            if not success:
                raise RuntimeError('video stream not available')
            resized = cv2.resize(frame, dsize=self._image_shape.wh, interpolation=cv2.INTER_CUBIC)
            assert resized.shape[2] == 3, 'image should be 3 channel'
            obs['rgb'] = resized
        return obs

    async def run(self):
        assert self._target_update_delta_time and self._command_delta_time, \
            'both target_update_delta_time and command_delta_time should be specified'
        last_pos = None
        while True:
            target_pos = None if self._last_target_position is None else np.copy(self._last_target_position)
            async with move_on_when(self._target_update_event.wait):
                start_at = trio.current_time()
                async for _ in periodic(self._command_delta_time):
                    with trio.CancelScope(shield=True):
                        if target_pos is not None:
                            if last_pos is None:
                                next_xy = target_pos
                            else:
                                delta = trio.current_time() - start_at
                                s = np.clip(delta / self._target_update_delta_time, 0, 1)
                                next_xy = s * target_pos + (1 - s) * last_pos
                            await self._move_end_effector_to(next_xy)
                        self._last_observation = await self._fetch_observation()
                        self._observation_update_event.set()
            last_pos = target_pos


@asynccontextmanager
async def open_arm_control(serial_port_name: str,
                           target_update_delta_time: float,
                           command_delta_time: float,
                           observations: Optional[Sequence] = None,
                           image_shape: Optional[ImageShape] = None):
    async with SerialStream(serial_port_name, baudrate=115200) as rs485_serial, \
            trio.open_nursery() as nursery:
        with (open_video_capture(0) if image_shape else nullcontext()) as cap:
            infra = RobotArmRealInfra(
                rs485_serial, cap, target_update_delta_time, command_delta_time, observations, image_shape)
            await infra._reset_pid_params()
            try:
                nursery.start_soon(infra.run)
                yield infra
            finally:
                nursery.cancel_scope.cancel()
                with trio.CancelScope(shield=True):
                    await trio.sleep(0.1)
                    await infra._shutdown()
