import functools
import typing
import numpy as np

from random import choices
from tqdm import tqdm

from helpers.functions import RosenbrockFitnessFunction
from helpers.models.optimization_settings import OptimizationSettings
from helpers.models.solution import Solution
from helpers.optimization import predict_and_update_scores_for_solutions


class GeneticAlgorithm:
    
    def __init__(self, optimization_settings: OptimizationSettings) -> None:
        self.optimization_settings = optimization_settings
    
    def run(
        self,
        calculate_fitness: typing.Callable,
        population: typing.List[Solution],
        iteration_callbacks: typing.List[typing.Callable] = []
    ) -> typing.List[Solution]:
        
        calculate_fitness(population)
        population.sort(key=functools.cmp_to_key(Solution.compare_solutions))
        population = population[0:self.optimization_settings.population_size]
        
        for _ in tqdm(range(self.optimization_settings.iterations_count)):
            new_solutions = []
            
            for _ in range(self.optimization_settings.recombinations_per_iteration):
                new_solution: Solution = Solution.recombination(*choices(population, k=2))
                new_solution.mutate()
                new_solutions.append(new_solution)
            
            calculate_fitness(new_solutions)  
            population += new_solutions
            population.sort(key=functools.cmp_to_key(Solution.compare_solutions))
            
            for callback in iteration_callbacks:
                population = callback(population)
            
            if population is None:
                return None
        
            population = population[0:self.optimization_settings.population_size]
                
        return population
    
    
class GeneticOptimization(GeneticAlgorithm):
    
    @staticmethod
    def calculate_average_score(population: typing.List[Solution]):
        total = sum([solution.value for solution in population])
        return total / len(population)
    
    def optimization(self, model, initial_solutions: typing.List[Solution] = []) -> typing.List[Solution]:
        
        def calculate_fitness(population: typing.List[Solution]) -> None:
            predict_and_update_scores_for_solutions(model, population)

        initial_population = [Solution.generate_random_solution() for _ in range(self.optimization_settings.initial_population_size)] + initial_solutions
        return [self.run(calculate_fitness, initial_population)[0]]


class GeneticTargetOptimization(GeneticAlgorithm):
    
    def optimization(self, target_score: float) -> typing.List[int]:
        
        def calculate_fitness(population: typing.List[Solution]) -> None:
            for solution in population:
                solution.value = RosenbrockFitnessFunction.calculate(np.array(solution.properties))
        
        self.target_result = None
        def check_for_target_score_callback(population: typing.List[Solution]) -> typing.List[Solution]:
            for solution in population:
                if solution.value <= target_score:
                    self.target_result = [int(solution.id), Solution.cid, RosenbrockFitnessFunction.run_count]
                    return None
            return population
        
        initial_population = [Solution.generate_random_solution() for _ in range(self.optimization_settings.initial_population_size)]
        population = self.run(calculate_fitness, initial_population, [check_for_target_score_callback])
        
        if population is not None:
            print("END", population[0].value, Solution.cid, RosenbrockFitnessFunction.run_count)
        return self.target_result
    
    
class GeneticGeneration(GeneticAlgorithm):
    
    def optimization(self, initial_solutions: typing.List[Solution] = []) -> typing.List[Solution]:
        
        def calculate_fitness(population: typing.List[Solution]) -> None:
            for solution in population:
                solution.value = RosenbrockFitnessFunction.calculate(np.array(solution.properties))
        
        best_solutions = []
        def take_best_solution_callback(population: typing.List[Solution]) -> typing.List[Solution]:
            best_solutions.append(population[0])
            population = population[1:]
            return population
        
        initial_population = [Solution.generate_random_solution() for _ in range(self.optimization_settings.initial_population_size)] + initial_solutions
        self.run(calculate_fitness, initial_population, [take_best_solution_callback])
        return best_solutions