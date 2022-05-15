import typing
import numpy as np
import statistics

from matplotlib import pyplot as plt
from tqdm import tqdm

import config.settings as settings

from helpers.file import read_model, read_population_from_file
from helpers.functions import rosenbrock
from helpers.model import predict
from helpers.models import Solution


def monte_carlo_method(model, max_score, iterations) -> typing.List[typing.List]:
    print(f"Monte Carlo testing with random generated networks on {iterations} iterations")
    
    solutions = [Solution.generate_random_solution() for _ in tqdm(range(iterations))] 
    predicted_scores = predict(model, solutions, max_score)
    
    results = []
    for index, solution in enumerate(solutions):
        results.append([rosenbrock(np.array(solution.properties)), predicted_scores[index]])
    return results


def main():
    model = read_model()
    
    _, scores = read_population_from_file()
    scores = [score[0] for score in scores]

    data = monte_carlo_method(model, max(scores), settings.MONTE_CARLO_TEST_POPULATION_SIZE)
    differances = [round(x[1] / x[0], 2) for x in data]
    median, stdev = round(statistics.median(differances), 2), round(statistics.stdev(differances), 2)
    print(f"Difference between surrogate predicted score and actual score median is {median} with standard deviation {stdev}")

    plt.hist(differances, bins=40)
    plt.show()


if __name__ == "__main__":
    main()