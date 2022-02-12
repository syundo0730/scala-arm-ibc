from math import cos, sin, atan2, acos, sqrt, pi
from typing import NamedTuple, Tuple, Collection
import numpy as np

_L0 = 0.196
_L1 = 0.196
_EPS = 1e-10  # small value to avoid zero division


def forward(angle_0: float, angle_1: float) -> Tuple[float, float]:
    """calc forward kinematics
    :param angle_0: 1st (root) joint angle
    :param angle_1: 2nd joint angle
    :return: position x, y
    """
    return (
        _L0 * cos(angle_0) + _L1 * cos(angle_0 + angle_1),
        _L0 * sin(angle_0) + _L1 * sin(angle_0 + angle_1),
    )


def inverse(point: Collection) -> Tuple[float, float]:
    """calc inverse kinematics
    :param point: position x, y
    :return: angles of arm joints
    """
    x, y = point
    squared = x**2 + y**2
    cos_val = (_L0**2 - _L1**2 + squared) / (2 * _L0 * sqrt(squared) + _EPS)
    cos_val = np.clip(cos_val, -1, 1)
    angle_0 = atan2(y, x) - acos(cos_val)
    cos_value = (_L0**2 + _L1**2 - squared) / (2 * _L0 * _L1)
    angle_1 = (pi - acos(cos_value)) if abs(cos_value) <= 1 else 0
    return angle_0, angle_1
