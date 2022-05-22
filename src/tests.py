import typing
import numpy as np
import statistics

from matplotlib import pyplot as plt
from tqdm import tqdm

import config.settings as settings

from helpers.file import read_model
from helpers.functions import rosenbrock
from helpers.model import predict
from helpers.models import Solution


def monte_carlo_method(model, iterations: int = settings.MONTE_CARLO_TEST_POPULATION_SIZE) -> typing.List[typing.List]:
    print(f"Monte Carlo testing with random generated networks on {iterations} iterations")
    
    solutions = [Solution.generate_random_solution() for _ in tqdm(range(iterations))] 
    predicted_scores = predict(model, solutions)
    
    results = []
    for index, solution in enumerate(solutions):
        results.append([rosenbrock(np.array(solution.properties)), predicted_scores[index]])
    return results


def main():
    model = read_model()

    data = monte_carlo_method(model)
    differances = [round((x[1] / x[0]) ** 1, 2) for x in data]
    median, stdev = round(statistics.median(differances), 2), round(statistics.stdev(differances), 2)
    print(f"Difference between surrogate predicted score and actual score median is {median} with standard deviation {stdev}")

    plt.hist(differances, bins=40)
    plt.show()


if __name__ == "__main__":
    main()