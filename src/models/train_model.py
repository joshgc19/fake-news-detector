# -*- coding: utf-8 -*-
"""
This file contains the function needed to train the Fake News Recognizer.
"""

from sklearn.tree import DecisionTreeClassifier


def train_model(x, y):
    """
    Function that trains a Decision Tree classifier with the given data
    Args:
        x(list): Features matrix
        y(list): Labels matrix

    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained model
    """
    df = DecisionTreeClassifier()
    df.fit(x, y)
    return df

