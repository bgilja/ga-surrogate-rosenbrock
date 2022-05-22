import numpy as np
import typing

from copy import deepcopy
from random import shuffle
from tensorflow import keras

from config import settings
from helpers.models import Solution
from helpers import scalers


def construct_model() -> keras.models.Sequential:
    # regularization = keras.regularizers.l2(settings.TRAINING_L2_REGULARIZATION)
    regularization = keras.regularizers.l1(0.1)
    input_shape = (settings.DIMENSIONS,)
    
    return keras.models.Sequential([
        keras.layers.Dense(512, activation="relu", kernel_regularizer=regularization, input_shape=input_shape),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=regularization),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=regularization),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=regularization),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=regularization),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(1, activation="relu")
    ])
    

def train_model(
        model: keras.models.Sequential,
        location_data: typing.List[typing.List[float]],
        scores: typing.List[typing.List[float]], 
        verbose: bool = True
    ) -> keras.models.Sequential:
    
    validation_split_percentage = 0.15
    validation_items = round(len(location_data) * validation_split_percentage)
    
    shuffle(location_data)
    shuffle(scores)

    X, Y = location_data[:-validation_items], scores[:-validation_items]
    X_val, Y_val = location_data[-validation_items:], scores[-validation_items:]
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=int(1e3), decay_rate=0.96, staircase=True)
    # lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(1e-4, int(1e2), t_mul=2.2, m_mul=1.0, alpha=0.0001)
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=settings.TRAINING_LOSS,
        metrics=settings.TRAINING_METRICS
    )
    
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

    history = model.fit(
        np.array(X),
        np.array(Y),
        validation_data=(X_val, Y_val),
        shuffle=True,
        verbose=verbose,
        batch_size=settings.TRAINING_BATCH_SIZE,
        epochs=settings.TRAINING_EPOCHS,
        callbacks=[early_stopping_callback]
    )
    
    return model, history


def predict(model, solutions: typing.List[Solution]) -> typing.List[float]:
    data = np.array([solution.properties for solution in solutions])
    predicted_values = np.array([x[0] for x in scalers.normalizer(model.predict(data))])
    return predicted_values


def continue_training(model, location_data, scores):
    # model = construct_model()
    # model, _ = train_model(model, location_data, scores, False)
    
    validation_split_percentage = 0.35
    validation_items = round(len(location_data) * validation_split_percentage)
    
    location_data_shuffled = deepcopy(location_data)
    shuffle(location_data_shuffled)
    scores_shuffled = deepcopy(scores)
    shuffle(scores_shuffled)
    
    X, Y = location_data_shuffled[:-validation_items], scores_shuffled[:-validation_items]
    X_val, Y_val = location_data_shuffled[-validation_items:], scores_shuffled[-validation_items:]
    
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    
    _ = model.fit(
        np.array(X),
        np.array(Y),
        validation_data=(np.array(X_val), np.array(Y_val)),
        shuffle=True,
        verbose=False,
        batch_size=settings.CONTINUE_TRAINING_BATCH_SIZE,
        epochs=settings.CONTINUE_TRAINING_EPOCHS,
        callbacks=[early_stopping_callback]
    )
    
    return model