from typing import NamedTuple, Tuple, Optional

import cv2
import gin
import numpy as np


class ImageShape(NamedTuple):
    w: int
    h: int
    channel: int

    def apply_wh(self, image: np.ndarray):
        return cv2.resize(image, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

    @property
    def np_array_shape(self) -> Tuple:
        return self.h, self.w, self.channel


class ImageCrop(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image[self.y:self.y+self.h, self.x:self.x+self.w]


@gin.configurable
def apply_crop_and_reshape(
        image: np.ndarray,
        image_crop: Optional[Tuple[float, float, float, float]] = None,
        image_shape: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    image_crop = ImageCrop(*image_crop)
    image_shape = ImageShape(*image_shape)
    return image_shape.apply_wh(image_crop.apply(image))
