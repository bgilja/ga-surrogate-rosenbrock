import typing

from random import choice, randrange, uniform

import numpy as np

import config.settings as settings

from helpers.functions import BaseFitnessFunction, FakeFitnessFunction


TSolution = typing.TypeVar("TSolution", bound="Solution")

class Solution:
    
    cid = 1
    solutions_count = settings.POPULATION_SIZE
    
    def __init__(self, properties: typing.List[float], value: typing.Union[None, float] = None) -> None:
        self.id = Solution.cid + 1
        Solution.cid += 1
        Solution.solutions_count += 1
        
        self.properties = properties
        self.value = value
      
    def __str__(self) -> str:
        return " ".join([str(v) for v in self.properties + [self.value]]) 
    
    def mutate(self) -> None:
        property_index = randrange(settings.DIMENSIONS)
        
        # value = (self.properties[property_index] + uniform(-1, 1)) * uniform(0.7, 1.3)
        value = self.properties[property_index] * uniform(0.9, 1.1)
        self.properties[property_index] = Solution.normalize_property(value)
        
    @staticmethod   
    def compare_solutions(s1: TSolution, s2: TSolution):
        return s1.value - s2.value
        
    @staticmethod  
    def generate_random_property():
        return round(uniform(*settings.BOUNDS), settings.DECIMAL_ROUNDING)
    
    @staticmethod  
    def generate_random_solution(fitness_function: BaseFitnessFunction = FakeFitnessFunction) -> TSolution:
        solution = Solution([Solution.generate_random_property() for _ in range(settings.DIMENSIONS)])
        solution.value = fitness_function.calculate(np.array(solution.properties))
        return solution
    
    @staticmethod  
    def generate_best_solution(fitness_function: BaseFitnessFunction = FakeFitnessFunction) -> TSolution:
        solution = Solution([1.0] * settings.DIMENSIONS)
        solution.value = fitness_function.calculate(np.array(solution.properties))
        return solution
    
    @staticmethod
    def normalize_property(value: float) -> float:
        if value == 0.0:
            value = 1 / (10 ** settings.DECIMAL_ROUNDING)
        if value > settings.BOUNDS[1]:
            return settings.BOUNDS[1]
        
        if value < settings.BOUNDS[0]:
            return settings.BOUNDS[0]
        
        return round(value, settings.DECIMAL_ROUNDING)
    
    @staticmethod
    def recombination(parent1: TSolution, parent2: TSolution) -> TSolution:
        if settings.DIMENSIONS < max(settings.CROSSOVER_RANGE):
            delimiter = randrange(0, settings.DIMENSIONS + 1)
        else:
            delimiter = randrange(settings.CROSSOVER_RANGE[0], settings.CROSSOVER_RANGE[1] + 1)
        return Solution(parent1.properties[:delimiter] + parent2.properties[delimiter:])