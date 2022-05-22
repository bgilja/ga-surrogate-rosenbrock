from matplotlib import pyplot

from config import settings
from helpers.file import read_population_from_file
from helpers.model import construct_model, train_model


def train():
    location_data, scores = read_population_from_file()
    
    model = construct_model()
    model, history = train_model(model, location_data, scores, False)
    
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    model.save(settings.MODEL_PATH, save_format=settings.MODEL_SAVE_FORMAT)
    
    return model


def main():
    train()


if __name__ == "__main__":
    main()