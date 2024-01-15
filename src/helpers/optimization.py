import typing

from helpers.model import predict
from helpers.models.solution import Solution


def predict_and_update_scores_for_solutions(model, solutions: typing.List[Solution]) -> None:
    predicted_scores = predict(model, solutions)
    for index, solution in enumerate(solutions):
        solution.value = predicted_scores[index]