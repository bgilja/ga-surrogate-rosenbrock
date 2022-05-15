import functools
import typing
import numpy as np

from random import choice
from tqdm import tqdm

import config.settings as settings

from helpers.file import read_model, read_population_from_file
from helpers.functions import rosenbrock
from helpers.model import predict
from helpers.models import Solution, compare_solutions


def predict_and_update_scores_for_solutions(model, solutions: typing.List[Solution], max_score: float) -> None:
    predicted_scores = predict(model, solutions, max_score)
    for index, solution in enumerate(solutions):
        solution.value = predicted_scores[index]
    

def optimize(model, max_score) -> Solution:
    population = [Solution.generate_random_solution() for _ in range(settings.OPTIMIZATION_INITIAL_POPULATION_SIZE)]
    predict_and_update_scores_for_solutions(model, population, max_score)
    
    for _ in tqdm(range(settings.OPTIMIZATION_ITERATIONS)):
        
        new_solutions = []
        for _ in range(settings.OPTIMIZATION_POPULATION_SIZE_PER_ITERATION):
            parents = [choice(population), choice(population)]
            new_solution: Solution = Solution.recombination(*parents)
            new_solution.mutate()
            new_solutions.append(new_solution)
        
        predict_and_update_scores_for_solutions(model, new_solutions, max_score)    
        population += new_solutions
        
        population.sort(key=functools.cmp_to_key(compare_solutions))
        population = population[:settings.OPTIMIZATION_INITIAL_POPULATION_SIZE]
        
    return population[0]


def main():
    model = read_model()

    _, scores = read_population_from_file()
    scores = [score[0] for score in scores]
    max_score = max(scores)

    solution: Solution = optimize(model, max_score)
    optimized_score = rosenbrock(np.array(solution.properties))
    
    random_solution = Solution.generate_random_solution()
    predict_and_update_scores_for_solutions(model, [random_solution], max_score)
    
    print(f"Random solution score: {random_solution.value}")
    print(f"Properties: {random_solution.properties}")
    
    print(f"Predicted score: {solution.value} | optimized score: {optimized_score}")
    print(f"Properties: {solution.properties}")


if __name__ == "__main__":
    main()