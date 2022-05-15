import typing
import numpy as np

import config.settings as settings
from helpers.algorithms import GeneticOptimization

from helpers.file import read_model, read_population_from_file
from helpers.functions import rosenbrock
from helpers.model import continue_training
from helpers.models import OptimizationSettings, Solution, calculate_rosenbrock_fitness
from helpers.optimization import predict_and_update_scores_for_solutions


def main():
    model = read_model()
    
    location_data, scores = read_population_from_file()
    
    random_solution = Solution.generate_random_solution(calculate_rosenbrock_fitness)
    print(f"Random solution rosenbrock score: {random_solution.value}")
    
    best_solution = Solution.generate_best_solution(calculate_rosenbrock_fitness)
    print(f"Best solution rosenbrock score: {best_solution.value}")
    
    predict_and_update_scores_for_solutions(model, [random_solution, best_solution])
    print(f"Random solution predicted score: {random_solution.value}")
    print(f"Best solution predicted score: {best_solution.value}")
    
    ga_settings = OptimizationSettings(
        settings.ITERATIVE_OPTIMIZATION_INITIAL_POPULATION_SIZE,
        settings.ITERATIVE_OPTIMIZATION_POPULATION_SIZE,
        settings.ITERATIVE_OPTIMIZATION_RECOMBINATIONS_PER_ITERATION,
        settings.ITERATIVE_OPTIMIZATION_ITERATIONS_PER_CYCLE,
    )
    ga_optimization = GeneticOptimization(ga_settings)
    
    new_scores = []
    for i in range(settings.ITERATIVE_OPTIMIZATION_CYCLES):
        print(f"Cycle #{i+1}: Generating optimized solutions")
        top_solutions: typing.List[Solution] = ga_optimization.optimization(model)
        
        top_location_data = []
        top_scores = []
        
        for solution in top_solutions:
            top_location_data.append(solution.properties)
            top_scores.append([rosenbrock(np.array(solution.properties))])
            
        print(f"Cycle #{i+1}: Best score {min(top_scores)} [predicted - {top_solutions[0].value}]")
        new_scores += top_scores
        
        location_data += top_location_data
        scores += top_scores
        
        if (i + 1) % settings.ITERATIVE_OPTIMIZATION_CONTINUE_TRAINING_AFTER_CYCLES == 0:
            print(f"#{i+1}: Continue training on existing model")
            model = continue_training(model, location_data, scores)
            
            predict_and_update_scores_for_solutions(model, [random_solution, best_solution])
            print(f"Random solution predicted score: {random_solution.value}")
            print(f"Best solution predicted score: {best_solution.value}")
            print(f"Best score found: {min(new_scores)[0]}")
    
    model.save(settings.MODEL_PATH, save_format=settings.MODEL_SAVE_FORMAT)


if __name__ == "__main__":
    main()