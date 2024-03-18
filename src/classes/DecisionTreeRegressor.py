
import numpy as np
import pandas as pd

from Node import Node


class DecisionTreeRegressor:

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, min_leaf=5):
        # Creates the root node, note: np.arrange with int parameter functions as range of python vanilla
        self.dtree: Node = Node(x, y, np.array(np.arange(len(y))), min_leaf)
        # Returns itself as this method also works as a constructor
        return self

    def predict(self, x: pd.DataFrame):
        return self.dtree.predict(x.to_numpy())
