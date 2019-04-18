"""
Problem B3: Principal Component Analysis of an Image
"""
import pandas as pd
from models.PCA import PCA
import numpy as np


if __name__ == '__main__':
    dims = [1, 2, 5, 10, 20, 30, 80, 100]
    for n_components in dims:
        feature_data = pd.read_csv("data/partB/boat-blocked.csv", header=None)
        pca = PCA(n_components)
        scores = pca.fit_transform(feature_data)
        reconstructed = pca.inverse_transform(scores)
        np.savetxt("data/partB/boat_{}_reconstructed.csv".format(n_components), reconstructed, delimiter=",")
        np.savetxt("data/partB/boat_{}_components.csv".format(n_components), pca.loadings, delimiter=",")
