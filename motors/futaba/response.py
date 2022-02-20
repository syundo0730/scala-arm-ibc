from typing import NamedTuple

_DATA_HEAD = 7


class PositionResponse(NamedTuple):
    angle: float

    @staticmethod
    def parse(data: bytes) -> 'PositionResponse':
        return PositionResponse(angle=int.from_bytes(data[_DATA_HEAD:_DATA_HEAD+2], 'little', signed=True) * .1)
