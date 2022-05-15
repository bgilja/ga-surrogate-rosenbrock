import functools
import typing

from random import choice
from tqdm import tqdm

from helpers.models import OptimizationSettings, Solution, compare_solutions
from helpers.optimization import predict_and_update_scores_for_solutions


class GeneticOptimization:
    
    def __init__(self, optimization_settings: OptimizationSettings) -> None:
        self.optimization_settings = optimization_settings
    
    def optimization(self, model) -> typing.List[Solution]:
        population = [Solution.generate_random_solution() for _ in range(self.optimization_settings.initial_population_size)]
        predict_and_update_scores_for_solutions(model, population)
        
        for _ in tqdm(range(self.optimization_settings.iterations_count)):

            new_solutions = []
            for _ in range(self.optimization_settings.recombinations_per_iteration):
                parents = [choice(population), choice(population)]
                new_solution: Solution = Solution.recombination(*parents)
                new_solution.mutate()
                new_solutions.append(new_solution)
            
            predict_and_update_scores_for_solutions(model, new_solutions)    
            population += new_solutions
            
            population.sort(key=functools.cmp_to_key(compare_solutions))
            population = population[:self.optimization_settings.population_size]
        
        return [population[0]]