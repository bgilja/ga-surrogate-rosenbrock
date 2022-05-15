import typing
from helpers.model import predict
from helpers.models import Solution


def predict_and_update_scores_for_solutions(model, solutions: typing.List[Solution], max_score: float) -> None:
    predicted_scores = predict(model, solutions, max_score)
    for index, solution in enumerate(solutions):
        solution.value = predicted_scores[index]