import functools
import typing

import config.settings as settings
from helpers.algorithms import GeneticGeneration

from helpers.file import save_population_to_file
from helpers.functions import RosenbrockFitnessFunction
from helpers.models.optimization_settings import OptimizationSettings
from helpers.models.solution import Solution
from helpers.visualize import visualize_data_properties, visualize_data_scores
    

def create_initial_population_random() -> typing.List[Solution]:
    population = [Solution.generate_random_solution(RosenbrockFitnessFunction) for _ in range(settings.POPULATION_SIZE)]
    population.sort(key=functools.cmp_to_key(Solution.compare_solutions))
    return population


def create_initial_population_ga() -> typing.List[Solution]:
    ga_settings = OptimizationSettings(
        settings.CREATE_INITIAL_POPULATION_SIZE,
        settings.POPULATION_SIZE,
        settings.CREATE_INITIAL_POPULATION_RECOMBINATIONS_PER_ITERATION,
        settings.CREATE_INITIAL_ITERATIONS,
    )
    ga_generation = GeneticGeneration(ga_settings)
    population = ga_generation.optimization()
    population.sort(key=functools.cmp_to_key(Solution.compare_solutions))
    return population
        
        
def main():
    if settings.GENERATE_INITIAL_POPULATION_RANDOM:
        population = create_initial_population_random()
    else:
        population = create_initial_population_ga()
    
    visualize_data_properties(population)
    visualize_data_scores(population)
    
    save_population_to_file(population)
        
        
if __name__ == "__main__":
    main()