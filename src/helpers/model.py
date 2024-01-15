import numpy as np
import typing

from copy import deepcopy
from tensorflow import keras

from config import settings
from helpers import scalers
from helpers.models.solution import Solution
from helpers.shuffle import shuffle_solutions
from helpers.types.neural_network_model import NeuralNetworkModel
from helpers.types.training_data import TrainingData
from helpers.visualize import visualize_validation_split


def construct_model(verbose: bool = True) -> NeuralNetworkModel:
    # regularization = keras.regularizers.l2(settings.TRAINING_L2_REGULARIZATION)
    regularization = keras.regularizers.l1(settings.TRAINING_L1_REGULARIZATION)
    
    nn_model = keras.models.Sequential([
        keras.layers.Dense(settings.TRAINING_HIDDEN_LAYER_SIZE, activation="relu", kernel_regularizer=regularization, input_shape=(settings.DIMENSIONS,)),
        # keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(settings.TRAINING_HIDDEN_LAYER_SIZE, activation="relu", kernel_regularizer=regularization),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(settings.TRAINING_HIDDEN_LAYER_SIZE, activation="relu", kernel_regularizer=regularization),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(settings.TRAINING_HIDDEN_LAYER_SIZE, activation="relu", kernel_regularizer=regularization),
        keras.layers.Dropout(settings.TRAINING_DROPOUT_RATE),
        keras.layers.Dense(1, activation="relu")
    ])
    
    if verbose:
        print(nn_model.summary())
        
    return nn_model
    
    
def continue_training(
    model: NeuralNetworkModel,
    data: TrainingData,
    verbose: bool = False
):  
    if settings.OPTIMIZATION_TRAIN_NEW_MODEL:
        model = construct_model(verbose=False)
        model, _ = train_model(model, data, verbose, 1000)
    else:
        validation_items = round(len(data[0]) * settings.CONTINUE_TRAINING_VALIDATION_SPLIT)
        location_data_shuffled, scores_shuffled = shuffle_solutions(deepcopy(data))
    
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=20,
            restore_best_weights=True
        )
        
        _ = model.fit(
            location_data_shuffled[:-validation_items],
            scores_shuffled[:-validation_items],
            validation_data=(
                location_data_shuffled[-validation_items:],
                scores_shuffled[-validation_items:]
            ),
            shuffle=True,
            verbose=False,
            batch_size=settings.CONTINUE_TRAINING_BATCH_SIZE,
            epochs=settings.CONTINUE_TRAINING_EPOCHS,
            callbacks=[early_stopping_callback]
        )
    
    return model


def predict(model, solutions: typing.List[Solution]) -> typing.List[float]:
    data = np.array([solution.properties for solution in solutions])
    prediction = model.predict(data)
    return np.array([x[0] for x in scalers.normalizer(prediction)])
    

def train_model(
    model: NeuralNetworkModel,
    data: TrainingData,
    verbose: bool = True,
    patience: int = 100
) -> NeuralNetworkModel:
    
    validation_items = round(len(data[0]) * settings.TRAINING_VALIDATION_SPLIT)
    location_data, scores = shuffle_solutions(data)
    
    location_data_train, location_data_validation = location_data[:-validation_items], location_data[-validation_items:]
    scores_train, scores_validation = scores[:-validation_items], scores[-validation_items:]
    
    visualize_validation_split(location_data_train, location_data_validation)

    initial_learning_rate = 5 * 1e-3
    final_learning_rate = 1e-6
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1 / settings.TRAINING_EPOCHS)
    steps_per_epoch = int(len(location_data_train) / settings.TRAINING_BATCH_SIZE)
    
    if verbose:
        print(f"Decay steps: {steps_per_epoch}")
        print(f"Decay rate: {learning_rate_decay_factor}")

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=steps_per_epoch,
                    decay_rate=learning_rate_decay_factor,
                    staircase=True)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss=settings.TRAINING_LOSS,
        metrics=settings.TRAINING_METRICS
    )
    
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        location_data_train,
        scores_train,
        validation_data=(
            location_data_validation,
            scores_validation
        ),
        shuffle=True,
        verbose=verbose,
        batch_size=settings.TRAINING_BATCH_SIZE,
        epochs=settings.TRAINING_EPOCHS,
        callbacks=[early_stopping_callback]
    )
    
    return model, history


def transform_solutions(solutions: typing.List[Solution]) -> TrainingData:
    location_data, scores = [], []
    
    for solution in solutions:
        location_data.append(solution.properties)
        scores.append([solution.value])

    return location_data, scores