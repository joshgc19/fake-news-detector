import numpy as np
import pandas as pd


class Node:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, indexes: np.ndarray, min_leaf: int = 5):
        self.x = x
        self.y = y
        self.indexes = indexes
        self.min_leaf = min_leaf
        self.row_count: int = len(indexes)
        # Column count can also be seen as features count
        self.col_count: int = x.shape[1]
        # Decision value which is the mean of the target
        self.val = np.mean(y[indexes])
        self.score = format('inf')
        # These 3 variables will be overwritten on the find_split_variable method
        self.split = -1
        self.lhs = []
        self.rhs = []

        self.find_split_variable()

    def find_split_variable(self):
        for col in range(self.col_count):
            self.find_better_split(col)

        if self.is_leaf:
            return

        x = self.split_col

        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]

        self.lhs = Node(self.x, self.y, self.indexes[lhs], self.min_leaf)
        self.rhs = Node(self.x, self.y, self.indexes[rhs], self.min_leaf)

    def find_better_split(self, var_idx: int):
        x = self.x.values[self.indexes, var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    @property
    def split_col(self):
        return self.x.values[self.indexes, self.var_idx]

    def predict(self, x):
        pass
