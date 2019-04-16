"""
Problem B2: Principal Component Analysis of Wines
"""
import numpy as np
import pandas as pd
from models.PCA import PCA


def get_dataset(name="red"):
    wines = pd.read_csv("data/partB/winequality-{}.txt".format(name), sep=" ", header=None)
    wines.columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                     "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    return wines


if __name__ == '__main__':
    pca = PCA(target_dims=2)
    wines = get_dataset()
    pca.fit(wines)
