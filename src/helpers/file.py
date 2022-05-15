import typing

from tensorflow import keras

import config.settings as settings

from helpers.models import Solution


def read_model():
    return keras.models.load_model(settings.MODEL_PATH)
      

def read_population_from_file() -> typing.Tuple[typing.List[typing.List[float]], typing.List[typing.List[float]]]:
    location_data = []
    scores = []
    
    with open(settings.POPULATION_PATH, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            values = [float(x) for x in line.split()]
            location_data.append(values[:-1])
            scores.append(values[-1:])
    
    return location_data, scores


def save_population_to_file(population: typing.List[Solution]):
    with open(settings.POPULATION_PATH, 'w', encoding='utf-8') as f:
        for solution in population:
            f.write(solution.get_export_data() + "\n")