import numpy as np

import config.settings as settings
from helpers.algorithms import GeneticOptimization

from helpers.file import read_model, read_population_from_file
from helpers.functions import RosenbrockFitnessFunction
from helpers.model import continue_training, transform_solutions
from helpers.models.optimization_settings import OptimizationSettings
from helpers.models.solution import Solution
from helpers.optimization import predict_and_update_scores_for_solutions
from helpers.visualize import plot


def main():
    model = read_model()
    
    solutions = list(read_population_from_file())
    training_data = transform_solutions(solutions)
    
    best_initial_score = min([score[0] for score in training_data[1]])
    
    random_solution = Solution.generate_random_solution(RosenbrockFitnessFunction)
    print(f"Random solution rosenbrock score: {random_solution.value}")
    
    best_solution = Solution.generate_best_solution(RosenbrockFitnessFunction)
    print(f"Best solution rosenbrock score: {best_solution.value}")
    
    RosenbrockFitnessFunction.run_count -= 2
    Solution.solutions_count -= 2
    
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
    best_scores = []
    optimized_solution = None
    best_optimized_solution = None
    
    for i in range(settings.ITERATIVE_OPTIMIZATION_CYCLES):
        print(f"Cycle {i+1} / {settings.ITERATIVE_OPTIMIZATION_CYCLES}: Generating optimized solutions")
        optimized_solution = ga_optimization.optimization(model, [] if optimized_solution is None else [optimized_solution])[0]
        # optimized_solution = ga_optimization.optimization(model, [] if optimized_solution is None else [])[0]
        
        if best_optimized_solution is None or best_optimized_solution.value > optimized_solution.value:
            best_optimized_solution = optimized_solution
        
        training_data[0].append(optimized_solution.properties)
            
        score = [RosenbrockFitnessFunction.calculate(np.array(optimized_solution.properties))]
        new_scores.append(score)
        training_data[1].append(score)
            
        print(f"Cycle {i+1} / {settings.ITERATIVE_OPTIMIZATION_CYCLES}: Best score {min(score)} [predicted - {optimized_solution.value}]")
        best_scores.append(min(new_scores))
        
        best_score = min(score[0] for score in new_scores)
        
        if (i + 1) % settings.ITERATIVE_OPTIMIZATION_CONTINUE_TRAINING_AFTER_CYCLES == 0:
            # print(f"#{i+1}: Continue training on existing model")
            model = continue_training(model, training_data)
            
            predict_and_update_scores_for_solutions(model, [random_solution, best_solution])
            print(f"Random solution predicted score: {random_solution.value}")
            print(f"Best solution predicted score: {best_solution.value}")
            print(f"Best initial score: {best_initial_score}")
            print(f"Best score found: {best_score}")
            print(f"Best score optimization percentage: {best_score / best_initial_score}")
    
    model.save(settings.MODEL_PATH, save_format=settings.MODEL_SAVE_FORMAT)
    
    plot([i+1 for i in range(settings.ITERATIVE_OPTIMIZATION_CYCLES)], [score[0] for score in best_scores], "Iteracija optimizacije", "Vrijednost Rosenbrock funkcije", best_initial_score)
    plot([i+1 for i in range(settings.ITERATIVE_OPTIMIZATION_CYCLES)], [score[0] for score in best_scores], "Iteracija optimizacije", "Vrijednost Rosenbrock funkcije", None)
    plot([i+1 for i in range(settings.ITERATIVE_OPTIMIZATION_CYCLES)], [score[0] for score in new_scores], "Iteracija optimizacije", "Vrijednost Rosenbrock funkcije", best_initial_score)
    plot([i+1 for i in range(settings.ITERATIVE_OPTIMIZATION_CYCLES)], [score[0] for score in new_scores], "Iteracija optimizacije", "Vrijednost Rosenbrock funkcije", None)
    print(f"Total rosenbrock function calls: {RosenbrockFitnessFunction.run_count}")
    print(f"Total solutions created: {Solution.solutions_count}")
    print(f"Best solution properties: {best_optimized_solution.properties}")


if __name__ == "__main__":
    main()