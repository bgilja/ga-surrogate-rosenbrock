from config import settings
from helpers.file import read_population_from_file
from helpers.model import construct_model, train_model, transform_solutions
from helpers.visualize import visualize_training_metric


def train():
    solutions = list(read_population_from_file())
    training_data = transform_solutions(solutions)
    
    model = construct_model(True)
    model, history = train_model(model, training_data, False)
    
    visualize_training_metric(history, "loss", ("Broj epohe", "Vrijednost funkcije gubitka"))
    # visualize_training_metric(history, "mean_absolute_error")
    # visualize_training_metric(history, "mean_absolute_percentage_error")
    # visualize_training_metric(history, "mean_squared_error")
    
    print(min(history.history["loss"]))
    model.save(settings.MODEL_PATH, save_format=settings.MODEL_SAVE_FORMAT)
    return model


def main():
    train()


if __name__ == "__main__":
    main()