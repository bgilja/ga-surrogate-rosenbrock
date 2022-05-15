import numpy as np
import tensorflow as tf
import typing

from config import settings
from helpers.models import Solution
from helpers import scalers


def predict(model, solutions: typing.List[Solution]) -> typing.List[float]:
    data = np.array([solution.properties for solution in solutions])
    predicted_values = np.array([x[0] for x in scalers.normalizer(model.predict(data))])
    return predicted_values


def continue_training(model, location_data, scores):
    data_train_in, data_train_out = np.array(location_data), np.array(scores)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    
    history = model.fit(
        data_train_in,
        data_train_out,
        shuffle=True,
        verbose=False,
        batch_size=settings.CONTINUE_TRAINING_BATCH_SIZE,
        epochs=settings.CONTINUE_TRAINING_EPOCHS,
        callbacks=[early_stopping_callback]
    )
    
    print(f"Continued training loss: {history.history['loss'][-1]}")
    
    return model