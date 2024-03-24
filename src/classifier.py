import scipy as sp
from sklearn.tree import DecisionTreeClassifier

from common.files_utils import load_csv_as_dataframe


def main():
    y = load_csv_as_dataframe("../../data/train.csv")["target"]
    x = sp.sparse.load_npz("../data/train_features_vectors.npz")

    dt = DecisionTreeClassifier()
    dt.fit(x, y)


if __name__ == "__main__":
    main()
