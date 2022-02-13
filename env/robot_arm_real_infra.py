import math
from contextlib import nullcontext, asynccontextmanager
from math import degrees, radians
from typing import List, Dict, Tuple, Optional, Set

import cv2
import numpy as np
import trio
from trio_serial import AbstractSerialStream, SerialStream

from camera.video_capture import open_video_capture
from core.serial_client import SerialClient
from env.arm_kinematics_2d import inverse, forward
from env.robot_infra import RobotArmInfra
from gyems.rmd_r485.commands import Commands


class RobotArmRealInfra(RobotArmInfra):
    _MOTOR_IDS = (1, 2)
    _JOINT_SPEED = 360  # degree / sec
    _COLLIDABLE_DISTANCE_TO_BASE = 0.1

    def __init__(self, serial_stream: AbstractSerialStream, video_stream: cv2.VideoCapture,
                 observations: Optional[Set], image_size: Optional[Tuple[float, float]]):
        self._serial_client = SerialClient(serial_stream)
        self._video_stream = video_stream
        self._observations = observations
        self._image_size = image_size

        self._last_target_position = np.zeros(2)

    @property
    def target_position(self) -> np.ndarray:
        return self._last_target_position

    async def _reset_pid_params(self):
        # write pid control params
        for id_ in self._MOTOR_IDS:
            await self._serial_client.command(Commands.write_control_param_ram(id_, 10, 100, 200, 30))

    async def _shutdown(self):
        for id_ in self._MOTOR_IDS:
            await self._serial_client.command(Commands.shutdown(id_))
            print(f'ID: {id_} shutdown')

    async def move_end_effector_to(self, xy: np.ndarray, joint_speed: Optional[float] = None) -> None:
        dist_from_origin = math.hypot(*xy)
        if dist_from_origin < self._COLLIDABLE_DISTANCE_TO_BASE:
            await trio.sleep(0)
            return
        self._last_target_position = xy
        angle_0, angle_1 = inverse(xy)
        # convert radians to degrees, and invert the sign (from ccw positive to cw positive)
        speed = degrees(joint_speed) if joint_speed is not None else self._JOINT_SPEED
        commands = [Commands.move_to_position_with_speed(id_, -degrees(angle), speed)
                    for id_, angle in zip(self._MOTOR_IDS, (angle_0, angle_1))]
        for command in commands:
            await self._serial_client.command(command)

    async def command_action(self, xy_diff: np.ndarray) -> None:
        new_target_position = self._last_target_position + xy_diff
        await self.move_end_effector_to(new_target_position)

    async def _read_joint_angles(self) -> List:
        commands = [Commands.read_position(id_) for id_ in self._MOTOR_IDS]
        # convert from degrees to radians, and invert the sign (from cw positive to ccw positive)
        return [-radians((await self._serial_client.query(command)).angle) for command in commands]

    async def get_observation(self) -> Dict:
        joint_angles = await self._read_joint_angles()
        end_effector_point = forward(*joint_angles)
        obs = {}
        if self._observations is None or 'joint_angles' in self._observations:
            obs['joint_angles'] = joint_angles
        if self._observations is None or 'end_effector_pos' in self._observations:
            obs['end_effector_pos'] = end_effector_point
        if (self._observations is None or 'rgb' in self._observations) and self._image_size is not None:
            success, frame = self._video_stream.read()
            if not success:
                raise RuntimeError('video stream not available')
            resized = cv2.resize(frame, dsize=self._image_size, interpolation=cv2.INTER_CUBIC)
            assert resized.shape[2] == 3, 'image should be 3 channel'
            obs['rgb'] = resized
        return obs


@asynccontextmanager
async def open_arm_control(serial_port_name: str, observations, image_size):
    async with SerialStream(serial_port_name, baudrate=115200) as rs485_serial:
        with (open_video_capture(0) if image_size else nullcontext) as cap:
            infra = RobotArmRealInfra(rs485_serial, cap, observations, image_size)
            await infra._reset()
            try:
                yield infra
            finally:
                with trio.CancelScope(shield=True):
                    await infra._shutdown()
