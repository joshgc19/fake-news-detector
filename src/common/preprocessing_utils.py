import re
import string
import pandas as pd
import math

from nltk import word_tokenize
from numpy import ndarray


def word_cleaning(text):
    text = text.lower()
    text = re.sub('[.*?]|\w*\d\w*|\n|https?:\\\S+|www\.\S+|<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\W', " ", text)
    return text


def clean_text(text: pd.DataFrame):
    # Apply a cleaning function to each row in a dataframe
    return text.apply(word_cleaning)


def split_data(matrix: pd.DataFrame, target_column: str, test_ratio: int = 0.25):
    matrix = matrix.sample(frac=1)
    test_matrix_length = math.floor(matrix.shape[0] * test_ratio)
    test_dataset = matrix[:test_matrix_length]
    train_dataset = matrix[test_matrix_length:]
    # return train_dataset[:, train_dataset.columns != target_column], train_dataset[target_column], test_dataset[:,
    #                                                                                                test_dataset.columns != target_column], \
    # test_dataset[target_column]
    return train_dataset, test_dataset



