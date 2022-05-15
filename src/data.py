import copy
import functools
import typing

from random import choice
from tqdm import tqdm

import config.settings as settings

from helpers.file import save_population_to_file
from helpers.models import Solution, calculate_rosenbrock_fitness, compare_solutions
    

def create_initial_population() -> typing.List[Solution]:
    population = [Solution.generate_random_solution(calculate_rosenbrock_fitness) for _ in range(settings.INITIAL_POPULATION_SIZE)]
    total_population = copy.deepcopy(population)
    
    for _ in tqdm(range(settings.ITERATIONS)):
        new_solutions = []
        for _ in range(settings.POPULATION_SIZE_PER_ITERATION):
            parents = [choice(population), choice(population)]
            new_solution: Solution = Solution.recombination(*parents)
            new_solution.mutate()
            new_solution.calculate_value(calculate_rosenbrock_fitness)
            new_solutions.append(new_solution)
        
        # new_solutions.sort(key=functools.cmp_to_key(compare_solutions))
        # new_solutions = [new_solutions[0]]
        
        population += new_solutions
        total_population += new_solutions
        
        population.sort(key=functools.cmp_to_key(compare_solutions))
        population = population[:settings.INITIAL_POPULATION_SIZE]
        
    total_population.sort(key=functools.cmp_to_key(compare_solutions))
    return total_population[-settings.POPULATION_SIZE:]
        
        
def main():
    population = create_initial_population()
    save_population_to_file(population)
        
        
if __name__ == "__main__":
    main()