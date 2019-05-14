import pandas as pd
from plotnine import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_dataset(name="red"):
    wines = pd.read_csv("data/partB/winequality-{}.txt".format(name), sep=" ", header=None)
    wines.columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                     "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    return wines


if __name__ == '__main__':
    wines = get_dataset("white")
    quality = pd.Series([])
    for i, row in wines.iterrows():
        if row['quality'] > 6:
            quality[i] = "high"
        elif row['quality'] == 6:
            quality[i] = 'medium'
        else:
            quality[i] = 'low'
    wines = wines.drop("quality", axis=1)
    lda = LinearDiscriminantAnalysis()
    scores = pd.DataFrame(lda.fit_transform(wines, quality), columns=["dim1", "dim2"])
    scores.insert(2, "quality", quality)
    scatter = (ggplot(aes(x="dim1", y="dim2"), data=scores)
               + geom_point(aes(color="factor(quality)"), alpha=0.4)
               + labs(x="DIM_1 ({:.2f} %)".format(lda.explained_variance_ratio_[0] * 100),
                      y="DIM_2 ({:.2f} %)".format(lda.explained_variance_ratio_[1] * 100),
                      color="quality")
               + ggtitle("white")
               + theme(subplots_adjust={'right': 0.80}))
    print(scatter)
