from random import shuffle

from helpers.types.training_data import TrainingData


def shuffle_solutions(t_data: TrainingData) -> TrainingData:
    data = list(zip(*t_data))
    shuffle(data)
    return list(zip(*data))