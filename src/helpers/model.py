import typing
import numpy as np

from tensorflow import keras

import config.settings as settings

from helpers.models import Solution
from helpers import scalers


def predict(model, solutions: typing.List[Solution], max_value: float) -> typing.List[float]:
    data = scalers.domain_scaler(np.array([solution.properties for solution in solutions]))
    predicted_values = model.predict(data)
    # return np.around(scalers.linear_inverse_scaler(predicted_values, max_value), settings.DECIMAL_ROUNDING)
    return np.around(scalers.logaritmic_inverse_scaler(predicted_values, max_value), settings.DECIMAL_ROUNDING)


def continue_training(model, location_data, scores, max_value):
    normalized_values = scalers.domain_scaler(location_data)
    normalized_scores = scalers.logaritmic_scaler(scores, max_value)
    
    data_train_in, data_train_out = normalized_values, normalized_scores
    
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1)
    
    model.fit(
        data_train_in,
        data_train_out,
        batch_size=100,
        shuffle=True,
        epochs=200,
        callbacks=[early_stopping_callback],
        verbose=True,
    )
    
    return model