import typing
import numpy as np

from config import settings


SCALE = 1. / (settings.BOUNDS[1] - settings.BOUNDS[0])
OFFSET = settings.BOUNDS[0]

def normalizer(values: np.ndarray) -> np.ndarray:
    return np.around(values, settings.DECIMAL_ROUNDING)

def domain_scaler(values: typing.List[float]) -> np.ndarray:
    return normalizer((np.array(values) - OFFSET) * SCALE)