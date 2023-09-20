import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, max_fitnesses, fully_parsed, title):
        self.max_fitnesses = max_fitnesses
        self.fully_parsed = fully_parsed
        self.title = str(title)

    def visualize(self):
        gens = len(self.max_fitnesses)
        xplot = np.array(list(range(gens)))
        yplot = np.array(self.max_fitnesses)
        plt.plot(xplot, yplot, color="red")
        plt.xlabel("Generation")
        plt.ylabel("Max Fitness")
        plt.title(self.title)
        plt.show()
        yplot = np.array(self.fully_parsed)
        plt.plot(xplot, yplot)
        plt.xlabel("Generation")
        plt.ylabel("Proportion Parsed Fully")
        plt.title(self.title)
        plt.show()
