"""
Problem Pre2: Curse of Dimensionality
"""
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
from plotnine import *
from sklearn.model_selection import train_test_split


def get_dataset(num_of_dims):
    ds = pd.DataFrame(np.random.normal(0, 1, (2000, num_of_dims)))
    column_names = []
    for i in range(num_of_dims):
        column_names.append("x{:}".format(i+1))
    ds.columns = column_names
    return ds


def target_function(x):
    return x + np.sin(5*x)


def k_nearest_neighbors(x_train, x_test, y_train, k):
    print("K-nearest neighbors (k={:}): predicting...".format(k))
    result = pd.DataFrame(columns=['y_pred'])
    i = 0
    for index, test_point in x_test.iterrows():
        neighbors_indexes = get_top5_neighbors(x_train, test_point, k)
        y_neighbors = y_train.iloc[neighbors_indexes]
        y_prediction = pd.DataFrame(data=[np.mean(y_neighbors)], index=[index], columns=['y_pred'])
        result = result.append(y_prediction)
        if i % 10 == 0:
            print("--> {:} %".format(i / 10))
        i += 1
    return result


def get_top5_neighbors(x_train, test_point, k):
    distances = []
    for index, train_point in x_train.iterrows():
        distances.append(euclidean(train_point, test_point))
    return sorted(range(len(distances)), key=lambda key: distances[key])[:k]


if __name__ == '__main__':
    dims = [1, 2, 4, 7, 10, 15]
    for dim in dims:
        ds = get_dataset(dim)
        y = target_function(ds['x1'])
        x_train, x_test, y_train, y_test = train_test_split(ds, y, test_size=0.5)
        y_test.name = "y_true"
        y_pred = k_nearest_neighbors(x_train, x_test, y_train, 5)
        x_test.to_pickle("../data/pre/{:}_x_test.pkl".format(dim))
        y_test.to_pickle("../data/pre/{:}_y_test.pkl".format(dim))
        y_pred.to_pickle("../data/pre/{:}_y_pred.pkl".format(dim))
        x_test = pd.read_pickle("../data/pre/{:}_x_test.pkl".format(dim))
        y_test = pd.read_pickle("../data/pre/{:}_y_test.pkl".format(dim))
        y_pred = pd.read_pickle("../data/pre/{:}_y_pred.pkl".format(dim))
        combined_test = pd.concat([x_test, y_test, y_pred], axis=1)
        print(combined_test.head())
        plot = (ggplot(aes(x='x1'), data=combined_test)
                + geom_point(mapping=aes(y='y_true'))
                + geom_point(mapping=aes(y='y_pred'), color="green", alpha=0.3)
                + labs(y="y", title="{:} dimensions".format(dim)))


