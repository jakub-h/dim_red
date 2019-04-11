"""
Problem Pre1: Properties of High-dimensional Spaces
"""
from scipy.spatial.distance import euclidean
import numpy as np


def get_dataset(num_of_points, num_of_dims):
    return 2 * np.random.random_sample((num_of_points, num_of_dims)) - 1


def get_distances(dataset):
    distances = []
    for point in dataset:
        distances.append(euclidean(point, np.zeros((num_of_dims, ))))
    return distances


if __name__ == '__main__':
    num_of_points = 100
    num_of_dims = 2
    dataset = get_dataset(num_of_points, num_of_dims)

