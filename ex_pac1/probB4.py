"""
Problem B4: Principal Component Analysis of an Audio File
"""
import pandas as pd
from models.PCA import PCA
import numpy as np
from plotnine import *


if __name__ == '__main__':
    feature_data = pd.read_csv("data/partB/entertainer_featuredata.csv", header=None)
    pca = PCA(15)
    scores = pca.fit_transform(feature_data)
    reconstructed = pca.inverse_transform(scores)
    np.savetxt("data/partB/entertainer_reconstructed.csv", reconstructed, delimiter=",")

    reconst_vec = pd.read_csv("data/partB/entertainer_reconstvector.csv", header=None, names=["v0"])
    vec = (ggplot(data=reconst_vec, mapping=aes(x=reconst_vec.index.values, y=reconst_vec["v0"].values)) +
           geom_line())
    print(vec)
