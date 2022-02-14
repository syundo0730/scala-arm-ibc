from typing import NamedTuple, Tuple


class ImageShape(NamedTuple):
    w: int
    h: int
    channel: int

    @property
    def wh(self) -> Tuple:
        return self.w, self.h

    @property
    def np_array_shape(self) -> Tuple:
        return self.h, self.w, self.channel
