"""
Problem Pre1: Properties of High-dimensional Spaces
"""
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
from plotnine import *


def get_dataset(num_of_points, num_of_dims):
    return 2 * np.random.random_sample((num_of_points, num_of_dims)) - 1


def get_distances(dataset):
    distances = []
    for point in dataset:
        distances.append(euclidean(point, np.zeros((dataset.shape[1], ))))
    return distances


def task_a():
    print("Task a)")
    dimensions = [2, 3, 5, 7, 10, 13, 17]
    num_of_points = 10000000
    for d in dimensions:
        distances = get_distances(get_dataset(num_of_points, d))
        inside = 0
        outside = 0
        for dist in distances:
            if dist > 1:
                outside += 1
            else:
                inside += 1
        print("--> d =", d)
        print("--> ratio =", inside/(outside + inside))
        with open("../data/pre/a", "a") as output:
            output.write("{:}\t{:}\n".format(d, inside/(inside + outside)))


def task_b():
    print("Task b)")
    dimensions = [2, 3, 5, 7, 10, 13, 17]
    num_of_points = 10000000
    for d in dimensions:
        distances = get_distances(get_dataset(num_of_points, d))
        inside = 0
        outside = 0
        shell = 0
        for dist in distances:
            if dist > 1:
                outside += 1
            else:
                inside += 1
                if dist >= 0.95:
                    shell += 1
        print("--> d =", d)
        print("--> ratio = ", shell/inside)
        with open("../data/pre/b", "a") as output:
            output.write("{:}\t{:}\n".format(d, shell/inside))


if __name__ == '__main__':
    # task_a()
    ratios = pd.read_csv("../data/pre/a", sep="\t", header=None)
    ratios.columns = ["d", "inside/all"]
    print(ratios)
    plot = (ggplot(aes("d", "inside/all"), data=ratios)
            + geom_point()
            + geom_line()
            + ggtitle("Pre: Task a)"))
    print(plot)

    # task_b()
    ratios = pd.read_csv("../data/pre/b", sep="\t", header=None)
    ratios.columns = ["d", "shell/inside"]
    print(ratios)
    plot = (ggplot(aes("d", "shell/inside"), data=ratios)
            + geom_point()
            + geom_line()
            + ggtitle("Pre: Task b)"))
    print(plot)

