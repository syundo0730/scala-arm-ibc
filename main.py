from typing import Sequence, List, NamedTuple, Optional

from gym import register
import trio
from trio_serial import SerialStream, AbstractSerialStream
from trio_util import periodic

from controller.policy_executor import PolicyExecutor
from env.robot_arm_env import RobotArmEnv
from motors.gyems.rmd_r485 import Commands as RmdCommands
from motors.futaba.commands import Commands as FutabaCommands
from controller.oracle_recorder import OracleRecorder


register(
    id='ScalaArm-v0',
    entry_point=RobotArmEnv,
)


class _MotorConversion(NamedTuple):
    input_id: int
    output_id: int
    invert_sign: bool = False

    def convert(self, rs304_angle: float) -> Optional[float]:
        if abs(rs304_angle) < 180:
            return (-1 if self.invert_sign else 1) * rs304_angle
        return None


_MOTOR_CONVERSIONS = (
    _MotorConversion(input_id=49, output_id=1),
    _MotorConversion(input_id=9, output_id=2, invert_sign=True),
)


async def _read_input_angles(serial: AbstractSerialStream, ids: Sequence[int]) -> List[float]:
    angles = []
    for id_ in ids:
        read_pos_command = FutabaCommands.read_position(id_)
        await serial.send_all(read_pos_command.command_bytes)
        res = await serial.receive_some()
        angles.append(read_pos_command.response_parser(res).angle)
    return angles


async def _command_and_read_current_angles(
        serial: AbstractSerialStream, ids: Sequence[int], angles: Sequence[Optional[float]]) -> List[float]:
    current_angles = []
    for id_, angle in zip(ids, angles):
        commands = [RmdCommands.read_position(id_)]
        if angle is not None:
            commands.append(RmdCommands.move_to_position_with_speed(id_, angle, 360))
        for command in commands:
            await serial.send_all(command.command_bytes)
            res = await serial.receive_some()
            if command.response_parser:
                current_angles.append(command.response_parser(res).angle)
    return current_angles


async def _save_current_position_as_origin(serial: AbstractSerialStream, id_):
    commands = (
        RmdCommands.save_current_position_as_origin(id_),
        RmdCommands.shutdown(id_),
        RmdCommands.resume(id_),
        RmdCommands.read_position(id_),
    )
    for command in commands:
        await serial.send_all(command.command_bytes)
        res = await serial.receive_some()
        if command.response_parser:
            angle = command.response_parser(res).angle
            print('reset origin angle: ', angle)
            assert angle == 0


async def _probe_futaba_id(serial: AbstractSerialStream):
    for id_ in range(1, 128):
        read_pos_command = FutabaCommands.read_position(id_)
        await serial.send_all(read_pos_command.command_bytes)
        with trio.move_on_after(.1):
            await serial.receive_some()
            print('got response from id: ', id_)
            return
        print('timed out', id_)


async def _direct_connect_input_output(ttl_serial, rs485_serial):
    input_ids = [conv.input_id for conv in _MOTOR_CONVERSIONS]
    output_ids = [conv.output_id for conv in _MOTOR_CONVERSIONS]

    try:
        async for _ in periodic(.02):
            input_angles = await _read_input_angles(ttl_serial, input_ids)
            target_angles = [conv.convert(angle) for conv, angle in zip(_MOTOR_CONVERSIONS, input_angles)]
            current_angles = await _command_and_read_current_angles(rs485_serial, output_ids, target_angles)
    finally:
        with trio.CancelScope(shield=True):
            for id_ in output_ids:
                await rs485_serial.send_all(RmdCommands.shutdown(id_).command_bytes)
                await rs485_serial.receive_some()


async def _read_control_params(serial: AbstractSerialStream):
    for conv in _MOTOR_CONVERSIONS:
        command = RmdCommands.read_control_param(conv.output_id)
        await serial.send_all(command.command_bytes)
        print(command.response_parser(await serial.receive_some()), conv.output_id)


async def _write_control_params(serial: AbstractSerialStream, *args):
    for conv in _MOTOR_CONVERSIONS:
        command = RmdCommands.write_control_param_ram(
            conv.output_id, *args)
        await serial.send_all(command.command_bytes)
        res = await serial.receive_some()
        print('control param written', res)


async def main_master_slave():
    async with SerialStream('/dev/tty.usbserial-11110', baudrate=115200) as ttl_serial, \
            SerialStream('/dev/tty.usbserial-0001', baudrate=115200) as rs485_serial:
        await _write_control_params(rs485_serial, 10, 100, 200, 30)
        await _direct_connect_input_output(ttl_serial, rs485_serial)


async def main_old2():
    async with SerialStream('/dev/tty.usbserial-0001', baudrate=115200) as rs485_serial:
        rs485_commands = (
            RmdCommands.move_to_position_with_speed(1, 0, 90),
            RmdCommands.move_to_position_with_speed(2, 0, 90),
            # RmdCommands.move_to_position_with_speed(gyems_id, 0, 90),
            RmdCommands.read_position(1),
            RmdCommands.read_position(2),
        )
        for command in rs485_commands:
            await rs485_serial.send_all(command.command_bytes)
            res = await rs485_serial.receive_some()
            if command.response_parser:
                print(command.response_parser(res).angle)
        # await _save_current_position_as_origin(rs485_serial, 1)
        # await _save_current_position_as_origin(rs485_serial, 2)


async def main_old():
    # reset
    async with SerialStream('/dev/tty.usbserial-0001', baudrate=115200) as rs485_serial, \
            SerialStream('/dev/tty.usbserial-130', baudrate=115200) as ttl_serial:
        futaba_id = 9
        gyems_id = 1
        read_pos_command = FutabaCommands.read_position(futaba_id)
        while True:
            await ttl_serial.send_all(read_pos_command.command_bytes)
            res = await ttl_serial.receive_some()
            if read_pos_command.response_parser:
                response = read_pos_command.response_parser(res)
                if abs(response.angle) < 180:
                    rs485_commands = (
                        RmdCommands.move_to_position_with_speed(gyems_id, response.angle, 90),
                        RmdCommands.read_position(gyems_id),
                    )
                    for command in rs485_commands:
                        await rs485_serial.send_all(command.command_bytes)
                        res = await rs485_serial.receive_some()
                        if command.response_parser:
                            print(command.response_parser(res).angle)
            await trio.sleep(.05)

trio.run(main)
