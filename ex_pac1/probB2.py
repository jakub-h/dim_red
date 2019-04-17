"""
Problem B2: Principal Component Analysis of Wines
"""
import pandas as pd
from models.PCA import PCA
from plotnine import *


def get_dataset(name="red"):
    wines = pd.read_csv("data/partB/winequality-{}.txt".format(name), sep=" ", header=None)
    wines.columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                     "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    return wines


if __name__ == '__main__':
    for type in ["red", "white"]:
        wines = get_dataset(type)
        pca = PCA(target_dims=2)
        scores = pca.fit_transform(wines)
        quality = pd.Series([])
        for i, row in wines.iterrows():
            if row['quality'] > 6:
                quality[i] = "high"
            elif row['quality'] == 6:
                quality[i] = 'medium'
            else:
                quality[i] = 'low'
        scores.insert(2, "quality", quality)
        scatter = (ggplot(aes(x="pc1", y="pc2"), data=scores)
                   + geom_point(aes(color="factor(quality)"), alpha=0.4)
                   + labs(x="PC1 ({:.2f} %)".format(pca.expl_variance[0] * 100),
                          y="PC2 ({:.2f} %)".format(pca.expl_variance[1] * 100),
                          color="quality")
                   + ggtitle(type)
                   + theme(subplots_adjust={'right': 0.80}))
        print(scatter)

