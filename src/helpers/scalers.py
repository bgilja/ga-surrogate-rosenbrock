import math
import numpy as np
import typing

from config import settings

SCALE = 1. / (settings.BOUNDS[1] - settings.BOUNDS[0])
OFFSET = settings.BOUNDS[0]

def normalizer(values: np.ndarray) -> np.ndarray:
    return np.around(values, settings.DECIMAL_ROUNDING)

def domain_scaler(values: typing.List[float]) -> np.ndarray:
    return normalizer((np.array(values) - OFFSET) * SCALE)

def linear_scaler(values: typing.List[float], max_value: float) -> np.ndarray:
    return normalizer(np.array(values) / max_value)

def linear_inverse_scaler(values: np.ndarray, max_value: float) -> np.ndarray:
    return normalizer(np.array([value[0] for value in values]) * max_value)

def logaritmic_scaler(values: typing.List[float], max_value: float) -> np.ndarray:
    return normalizer(np.log(np.array(values) + 1) / np.log(max_value))

def logaritmic_inverse_scaler(values: typing.List[float], max_value: float) -> np.ndarray:
    max_log = math.log(max_value)
    return normalizer(np.array([math.e ** (value[0] * max_log) for value in values]) - 1)