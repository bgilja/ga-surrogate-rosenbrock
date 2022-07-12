import typing

from tensorflow import keras


NeuralNetworkModel = typing.TypeVar("NeuralNetworkModel", bound="keras.models.Sequential")