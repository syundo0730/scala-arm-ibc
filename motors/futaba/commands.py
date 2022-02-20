from functools import reduce

from motors.core.command import Command
from motors.futaba.response import PositionResponse

_COMMON_HEADER = bytes([0xFA, 0xAF])


class Commands:
    @staticmethod
    def read_position(motor_id: int) -> Command:
        body = bytes([
            motor_id,
            0x09,  # flag
            0x00,  # addr
            0x00,  # len
            0x01,  # cnt
        ])
        checksum = reduce(lambda a, b: a ^ b, body) & 0xFF
        command = _COMMON_HEADER + body + bytes([checksum])
        return Command(command, PositionResponse.parse)
