import functools
import typing
import numpy as np

from random import choice
from tqdm import tqdm

import config.settings as settings

from helpers.file import read_model, read_population_from_file
from helpers.functions import rosenbrock
from helpers.model import continue_training
from helpers.models import Solution, calculate_rosenbrock_fitness, compare_solutions
from helpers.optimization import predict_and_update_scores_for_solutions
    

def optimization_cycle(model, max_score) -> typing.List[Solution]:
    population = [Solution.generate_random_solution() for _ in range(settings.ITERATIVE_OPTIMIZATION_INITIAL_POPULATION_SIZE)]
    predict_and_update_scores_for_solutions(model, population, max_score)
    
    for _ in tqdm(range(settings.ITERATIVE_OPTIMIZATION_ITERATIONS_PER_CYCLE)):

        new_solutions = []
        for _ in range(settings.ITERATIVE_OPTIMIZATION_ITERATIONS_PER_CYCLE):
            parents = [choice(population), choice(population)]
            new_solution: Solution = Solution.recombination(*parents)
            new_solution.mutate()
            new_solutions.append(new_solution)
        
        predict_and_update_scores_for_solutions(model, new_solutions, max_score)    
        population += new_solutions
        
        population.sort(key=functools.cmp_to_key(compare_solutions))
        population = population[:settings.ITERATIVE_OPTIMIZATION_INITIAL_POPULATION_SIZE]
    
    return [population[0]]


def main():
    model = read_model()
    
    location_data, scores = read_population_from_file()
    max_score = max([score[0] for score in scores])
    
    random_solution = Solution.generate_random_solution(calculate_rosenbrock_fitness)
    print(f"Random solution rosenbrock score: {random_solution.value}")
    
    best_solution = Solution.generate_best_solution(calculate_rosenbrock_fitness)
    print(f"Best solution rosenbrock score: {best_solution.value}")
    
    predict_and_update_scores_for_solutions(model, [random_solution, best_solution], max_score)
    print(f"Random solution predicted score: {random_solution.value}")
    print(f"Best solution predicted score: {best_solution.value}")
    
    new_scores = []
    for i in range(settings.ITERATIVE_OPTIMIZATION_CYCLES): # TODO - mean squared error / absolute error
        print(f"Cycle #{i+1}: Generating optimized solutions")
        top_solutions: typing.List[Solution] = optimization_cycle(model, max_score)
        
        top_location_data = []
        top_scores = []
        
        for solution in top_solutions:
            top_location_data.append(solution.properties)
            top_scores.append([rosenbrock(np.array(solution.properties))])
            
        print(f"Cycle #{i+1}: Best score {min(top_scores)} [predicted - {top_solutions[0].value}]")
        new_scores += top_scores
        
        location_data += top_location_data
        scores += top_scores
        
        if (i + 1) % 5 == 0:
            print(f"#{i+1}: Continue training on existing model")
            model = continue_training(model, location_data, scores, max_score)
            
            predict_and_update_scores_for_solutions(model, [random_solution, best_solution], max_score)
            print(f"Random solution predicted score: {random_solution.value}")
            print(f"Best solution predicted score: {best_solution.value}")
    
    print(min(new_scores))
    model.save(settings.MODEL_PATH, save_format=settings.MODEL_SAVE_FORMAT)


if __name__ == "__main__":
    main()