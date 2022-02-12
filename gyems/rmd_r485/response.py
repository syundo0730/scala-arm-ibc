from typing import NamedTuple

_DATA_HEAD = 5


class PositionResponse(NamedTuple):
    angle: float

    @staticmethod
    def parse_multi_turn(data: bytes) -> 'PositionResponse':
        return PositionResponse(angle=int.from_bytes(data[_DATA_HEAD:_DATA_HEAD+8], 'little', signed=True) * .01)

    @staticmethod
    def parse_single_turn(data: bytes) -> 'PositionResponse':
        return PositionResponse(angle=int.from_bytes(data[_DATA_HEAD:_DATA_HEAD+2], 'little', signed=False) * .01)


class ControlParamResponse(NamedTuple):
    angle_control_p: int
    angle_control_i: int
    speed_control_p: int
    speed_control_i: int
    torque_control_p: int
    torque_control_i: int

    @staticmethod
    def parse(data: bytes) -> 'ControlParamResponse':
        params = data[_DATA_HEAD:_DATA_HEAD+6]
        return ControlParamResponse(
            angle_control_p=params[0],
            angle_control_i=params[1],
            speed_control_p=params[2],
            speed_control_i=params[3],
            torque_control_p=params[4],
            torque_control_i=params[5],
        )

    @staticmethod
    def parse_single_turn(data: bytes) -> 'PositionResponse':
        return PositionResponse(angle=int.from_bytes(data[_DATA_HEAD:_DATA_HEAD+2], 'little', signed=False) * .01)
