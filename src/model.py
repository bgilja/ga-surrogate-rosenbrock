from tensorflow import keras

from config import settings
from helpers.file import read_population_from_file
from helpers import scalers


def train_model():
    location_data, scores = read_population_from_file()

    scores = [score[0] for score in scores]
    normalized_values = scalers.domain_scaler(location_data)
    # normalized_scores = scalers.linear_scaler(scores, max(scores))
    normalized_scores = scalers.logaritmic_scaler(scores, max(scores))

    data_train_in, data_train_out = normalized_values, normalized_scores

    inputs = keras.Input(shape=(settings.DIMENSIONS,), name="input")
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense_3")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense_4")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_5")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_6")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=int(1e3), decay_rate=0.96, staircase=True)

    model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")

    model.fit(
        data_train_in,
        data_train_out,
        batch_size=100,
        shuffle=True,
        epochs=100,
        callbacks=[early_stopping_callback]
    )

    model.save(settings.MODEL_PATH, save_format=settings.MODEL_SAVE_FORMAT)
    return model


def main():
    train_model()


if __name__ == "__main__":
    main()