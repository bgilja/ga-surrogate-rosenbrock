from matplotlib import pyplot as plt
import numpy as np


def plot(x, y, axhline):
    _, ax = plt.subplots()
    
    max_value = max(max(y), axhline)
    xticks = np.around(np.linspace(x[0], x[-1], 10))
    
    ax.plot(x, y, linewidth=2.0)
    ax.set(xlim=(x[0], x[-1]), xticks=xticks, ylim=(0, max_value * 1.25))
    
    plt.axhline(y=axhline, color='r', linestyle='-')
    plt.show()