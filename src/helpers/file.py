import typing

from tensorflow import keras

import config.settings as settings

from helpers.models.solution import Solution
from helpers.types.neural_network_model import NeuralNetworkModel


def read_model() -> NeuralNetworkModel:
    return keras.models.load_model(settings.MODEL_PATH)
      

def read_population_from_file() -> typing.List[Solution]:
    with open(settings.POPULATION_PATH, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            values = [float(x) for x in line.split()]
            yield Solution(values[:-1], values[-1])


def save_population_to_file(population: typing.List[Solution]):
    with open(settings.POPULATION_PATH, 'w', encoding='utf-8') as f:
        for solution in population:
            f.write(f"{solution}\n")