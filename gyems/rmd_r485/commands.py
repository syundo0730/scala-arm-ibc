from typing import Optional

from core.command import Command
from gyems.rmd_r485.response import PositionResponse, ControlParamResponse

_COMMON_HEADER = 0x3E


def _compute_command_bytes(command_type: int, motor_id: int, *, body: Optional[bytes] = None):
    if body is None:
        body = bytes()

    frame_bytes = bytes([
        _COMMON_HEADER,
        command_type,
        motor_id,
        len(body),
    ])
    command_bytes = frame_bytes + bytes([sum(frame_bytes) & 0xFF])
    if body:
        command_bytes += (body + bytes([sum(body) & 0xFF]))
    return command_bytes  # limit each value in byte-size


class Commands:
    @staticmethod
    def read_control_param(motor_id: int) -> Command:
        return Command(
            command_bytes=_compute_command_bytes(0x30, motor_id),
            response_parser=ControlParamResponse.parse,
        )

    @staticmethod
    def write_control_param_ram(
            motor_id: int,
            angle_control_p: int = 100, angle_control_i: int = 100,
            speed_control_p: int = 40, speed_control_i: int = 30,
            torque_control_p: int = 50, torque_control_i: int = 50
    ) -> Command:
        return Command(_compute_command_bytes(
            0x31, motor_id, body=bytes([angle_control_p, angle_control_i,
                                        speed_control_p, speed_control_i,
                                        torque_control_p, torque_control_i])))

    @staticmethod
    def shutdown(motor_id: int) -> Command:
        return Command(_compute_command_bytes(0x80, motor_id))

    @staticmethod
    def pause(motor_id: int) -> Command:
        return Command(_compute_command_bytes(0x81, motor_id))

    @staticmethod
    def resume(motor_id: int) -> Command:
        return Command(_compute_command_bytes(0x88, motor_id))

    @staticmethod
    def save_current_position_as_origin(motor_id: int) -> Command:
        return Command(_compute_command_bytes(0x19, motor_id))

    @staticmethod
    def read_position(motor_id: int) -> Command:
        return Command(
            _compute_command_bytes(0x92, motor_id),
            PositionResponse.parse_multi_turn,
        )

    @staticmethod
    def read_single_loop_position(motor_id: int) -> Command:
        return Command(
            _compute_command_bytes(0x94, motor_id),
            PositionResponse.parse_single_turn,
        )

    @staticmethod
    def move_to_position(motor_id: int, target_degree: float) -> Command:
        deg_x100 = int(target_degree * 100).to_bytes(8, 'little', signed=True)
        return Command(_compute_command_bytes(0xA3, motor_id, body=deg_x100))

    @staticmethod
    def move_to_position_with_speed(motor_id: int, target_degree: float, max_speed: float) -> Command:
        # TODO: overrideできないのか
        deg_x100 = int(target_degree * 100).to_bytes(8, 'little', signed=True)
        speed_x100 = int(max_speed * 100).to_bytes(4, 'little', signed=True)
        return Command(_compute_command_bytes(0xA4, motor_id, body=deg_x100 + speed_x100))

    @staticmethod
    def move_to_single_loop_position_row(motor_id: int, target_degree: float, turn_clockwise: bool) -> Command:
        sign_flag = int(not turn_clockwise)  # direction 0x00: clock wise, 0x01 counter clock wise
        angles = int(target_degree * 100).to_bytes(2, 'little', signed=False)
        return Command(_compute_command_bytes(0xA5, motor_id, body=bytes([sign_flag]) + angles + bytes([0x00])))

    @staticmethod
    def get_model_info(motor_id: int) -> Command:
        return Command(_compute_command_bytes(0x12, motor_id))
