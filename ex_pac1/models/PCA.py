import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, target_dims):
        self.target_dims = target_dims
        self.scaler = StandardScaler()
        self.eig_vals = None
        self.eig_vecs = None
        self.expl_variance = None
        self.loadings = None

    def fit(self, x_train):
        x_std = StandardScaler().fit_transform(x_train)
        cov_mat = np.cov(x_std.T)
        self.eig_vals, self.eig_vecs = np.linalg.eig(cov_mat)
        sorted_indx = np.flip(np.argsort(self.eig_vals))
        self.eig_vecs = self.eig_vecs[sorted_indx]
        self.eig_vals = self.eig_vals[sorted_indx]
        self.expl_variance = []
        for eigval in self.eig_vals:
            self.expl_variance.append(eigval / self.eig_vals.sum())
        self.expl_variance = np.array(self.expl_variance)[:self.target_dims]
        self.eig_vecs = self.eig_vecs[:self.target_dims]
        self.eig_vals = self.eig_vals[:self.target_dims]
        self.loadings = np.ndarray(self.eig_vecs.shape)
        for i in range(self.eig_vecs.shape[0]):
            self.loadings[i] = self.eig_vecs[i] * np.sqrt(self.eig_vals[i])

    def transform(self, x):
        x_std = self.scaler.fit_transform(x)
        scores = pd.DataFrame(x_std.dot(self.eig_vecs.T))
        column_names = []
        for i in range(self.target_dims):
            column_names.append("pc{}".format(i+1))
        scores.columns = column_names
        return scores

    def fit_transform(self, x_train):
        self.fit(x_train)
        return self.transform(x_train)

    def inverse_transform(self, x):
        inverse = x.dot(self.eig_vecs)
        return pd.DataFrame(self.scaler.inverse_transform(inverse))




