import typing

from random import randrange, uniform
from typing import Callable, TypeVar

import numpy as np

import config.settings as settings
from helpers.functions import fake_fitness, rosenbrock

TSolution = TypeVar("TSolution", bound="Solution")

class Solution:
    
    def __init__(self, properties: typing.List[float]) -> None:
        self.properties = properties
        self.value = None
      
    @staticmethod  
    def generate_random_property():
        return round(uniform(*settings.BOUNDS), settings.DECIMAL_ROUNDING)
    
    @staticmethod  
    def generate_random_solution(calculate_fitness: Callable[[TSolution], float] = fake_fitness) -> TSolution:
        properties = [Solution.generate_random_property() for _ in range(settings.DIMENSIONS)]
        solution = Solution(properties)
        solution.value = calculate_fitness(solution)
        return solution
    
    @staticmethod  
    def generate_best_solution(calculate_fitness: Callable[[TSolution], float] = fake_fitness) -> TSolution:
        properties = [1.0 for _ in range(settings.DIMENSIONS)]
        solution = Solution(properties)
        solution.value = calculate_fitness(solution)
        return solution
    
    @staticmethod
    def recombination(parent1: TSolution, parent2: TSolution) -> TSolution:
        delimiter = randrange(settings.DIMENSIONS)
        return Solution(parent1.properties[:delimiter] + parent2.properties[delimiter:])
    
    def calculate_value(self, calculate_fitness: Callable[[TSolution], float]) -> None:
        self.value = calculate_fitness(self)
        
    def get_export_data(self):
        return " ".join([str(v) for v in self.properties + [self.value]])
    
    def mutate(self) -> None:
        property_index = randrange(settings.DIMENSIONS)
        self.properties[property_index] += (5 * uniform(-1, 1))
        self.properties[property_index] = normalize_property(self.properties[property_index])
        
        
def compare_solutions(s1: Solution, s2: Solution):
    return s1.value - s2.value

def calculate_rosenbrock_fitness(solution: Solution) -> float:
    return rosenbrock(np.array(solution.properties))

def normalize_property(value: float) -> float:
    if value > settings.BOUNDS[1]:
        return settings.BOUNDS[1]
    if value < settings.BOUNDS[0]:
        return settings.BOUNDS[0]
    return round(value, settings.DECIMAL_ROUNDING)


class OptimizationSettings:
    
    def __init__(
        self,
        initial_population_size,
        population_size,
        recombinations_per_iteration,
        iterations_count
    ) -> None:
        self.initial_population_size = initial_population_size
        self.population_size = population_size
        self.recombinations_per_iteration = recombinations_per_iteration
        self.iterations_count = iterations_count