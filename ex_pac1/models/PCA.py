import numpy as np
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, target_dims):
        self.target_dims = target_dims
        self.eig_vals = None
        self.eig_vecs = None

    def fit(self, x_train):
        x_std = StandardScaler().fit_transform(x_train)
        cov_mat = np.cov(x_std.T)
        self.eig_vals, self.eig_vecs = np.linalg.eig(cov_mat)
        sorted_indx = np.flip(np.argsort(self.eig_vals))
        self.eig_vecs = self.eig_vecs[sorted_indx][:self.target_dims]
        self.eig_vals = self.eig_vals[sorted_indx][:self.target_dims]
        print("Eigenvalues\n{}".format(self.eig_vals))
        print("Eigenvectors\n{}".format(self.eig_vecs))

    def transform(self, x):
        x_std = StandardScaler().fit_transform(x)



