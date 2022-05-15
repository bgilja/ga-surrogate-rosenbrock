import typing
import numpy as np

import config.settings as settings

def fake_fitness(_: typing.Any) -> float:
    return 0.0

def rosenbrock(x: np.ndarray) -> float:
    return round(float(np.sum(100 * (x.T[1:] - x.T[:-1] ** 2.0) ** 2 + (1 - x.T[:-1]) ** 2.0, axis=0)), settings.DECIMAL_ROUNDING)