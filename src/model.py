import numpy as np

from tensorflow import keras

from config import settings
from helpers.file import read_population_from_file


def train_model():
    location_data, scores = read_population_from_file()
    scores = [score[0] for score in scores]

    data_train_in, data_train_out = np.array(location_data), np.array(scores)

    inputs = keras.Input(shape=(settings.DIMENSIONS,), name="input")
    x = keras.layers.Dense(32, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense_3")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense_4")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense_5")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_6")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_7")(x)
    outputs = keras.layers.Dense(1, activation="relu", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=settings.TRAINING_OPTIMIZER,
        loss=settings.TRAINING_LOSS,
        metrics=settings.TRAINING_METRICS
    )

    model.fit(
        data_train_in,
        data_train_out,
        batch_size=settings.TRAINING_BATCH_SIZE,
        shuffle=True,
        epochs=settings.TRAINING_EPOCHS
    )

    model.save(settings.MODEL_PATH, save_format=settings.MODEL_SAVE_FORMAT)
    return model


def main():
    train_model()


if __name__ == "__main__":
    main()