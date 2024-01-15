import numpy as np

from abc import ABC, abstractmethod

import config.settings as settings
from helpers.types.fitness_properties import FitnessProperties


class BaseFitnessFunction(ABC):
    run_count = 0
    
    @staticmethod
    def calculate(values: FitnessProperties = None) -> float:
        pass
        

class FakeFitnessFunction(BaseFitnessFunction):
    
    @staticmethod
    def calculate(_: FitnessProperties = None) -> float:
        FakeFitnessFunction.run_count += 1
        return 0.0


class RosenbrockFitnessFunction(BaseFitnessFunction):
    run_count = settings.POPULATION_SIZE
    
    @staticmethod
    def calculate(values: FitnessProperties = None) -> float:
        RosenbrockFitnessFunction.run_count += 1
        return round(
            float(np.sum(100 * (values.T[1:] - values.T[:-1] ** 2.0) ** 2 + (1 - values.T[:-1]) ** 2.0, axis=0)),
            settings.DECIMAL_ROUNDING
        )