from turtle import position
import typing
import numpy as np

from matplotlib import pyplot as plt

from config import settings
from helpers.models.solution import Solution
from helpers.types.property import Property


def plot(x, y, label_x: str, label_y: str, axhline: Property = None):
    _, ax = plt.subplots()
    
    max_value = max(max(y), 0 if axhline is None else axhline)
    xticks = np.around(np.linspace(x[0], x[-1], 10))
    
    ax.plot(x, y, linewidth=2.0)
    ax.set(xlim=(x[0], x[-1]), xticks=xticks, ylim=(0, max_value * 1.25))
    
    if axhline is not None:
        plt.axhline(y=axhline, color='r', linestyle='-')
    
    plt.xlabel(label_x, fontsize=settings.PLOT_LABEL_FONT_SIZE)
    plt.ylabel(label_y, fontsize=settings.PLOT_LABEL_FONT_SIZE)
    plt.xticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.yticks(fontsize=settings.PLOT_TICKS_FONT_SIZE) 
    plt.show()
    

def visualize_data_properties(population: typing.List[Solution], color: str = None) -> None:
    if settings.DIMENSIONS != 2:
        return
    
    X = [solution.properties[0] for solution in population]
    Y = [solution.properties[1] for solution in population]
    
    _, ax = plt.subplots()
    ax.scatter(X, Y, c=color)
    
    plt.xlim(*settings.BOUNDS)
    plt.ylim(*settings.BOUNDS)
    plt.xticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.yticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.show()


def visualize_data_scores(population: typing.List[Solution]) -> None:
    _, ax = plt.subplots()
    
    X = [idx+1 for idx, _ in enumerate(population)]
    Y = [solution.value for solution in population]
    Y.sort()
    
    ax.bar(X, Y)
    plt.xticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.yticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.xlabel('Redni broj rješenja', fontsize=settings.PLOT_LABEL_FONT_SIZE)
    plt.ylabel('Vrijednost Rosenbrock funkcije', fontsize=settings.PLOT_LABEL_FONT_SIZE)
    plt.show()
    
    
def visualize_validation_split(location_data_train: typing.List[typing.List[float]], location_data_validation: typing.List[typing.List[float]]) -> None:
    if settings.DIMENSIONS != 2:
        return
    
    X_train = [solution_properties[0] for solution_properties in location_data_train]
    Y_train = [solution_properties[1] for solution_properties in location_data_train]
    
    X_validation = [solution_properties[0] for solution_properties in location_data_validation]
    Y_validation = [solution_properties[1] for solution_properties in location_data_validation]

    _, ax = plt.subplots()
    ax.scatter(X_train, Y_train)
    ax.scatter(X_validation, Y_validation, c="red")
    
    plt.xlim(*settings.BOUNDS)
    plt.ylim(*settings.BOUNDS)
    plt.xticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.yticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.show()  

def visualize_training_metric(history: typing.Any, metric: str, labels: typing.Tuple[str, str]) -> None:
    loss_metric = f"val_{metric}"
    plt.figure(figsize=(10, 8))
    plt.plot(history.epoch, history.history[metric], history.history[loss_metric])
    plt.legend([metric, loss_metric], prop={'size': 20})
    plt.xticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.yticks(fontsize=settings.PLOT_TICKS_FONT_SIZE)
    plt.xlabel(labels[0], fontsize=settings.PLOT_LABEL_FONT_SIZE)
    plt.ylabel(labels[1], fontsize=settings.PLOT_LABEL_FONT_SIZE)
    plt.show()