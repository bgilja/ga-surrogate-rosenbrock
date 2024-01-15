import config.settings as settings
from helpers.algorithms import GeneticTargetOptimization

from helpers.models.optimization_settings import OptimizationSettings


def find_target(target_score: float) -> int:
    ga_settings = OptimizationSettings(
        settings.TARGET_OPTIMIZATION_POPULATION_SIZE,
        settings.POPULATION_SIZE,
        settings.TARGET_OPTIMIZATION_POPULATION_RECOMBINATIONS_PER_ITERATION,
        settings.TARGET_OPTIMIZATION_ITERATIONS,
    )
    ga_generation = GeneticTargetOptimization(ga_settings)
    return ga_generation.optimization(target_score)


def main():
    # print(find_target(1.28))
    # print(find_target(47.823))
    # print(find_target(5226245.271))
    # print(find_target(608669.525))
    pass


if __name__ == "__main__":
    main()