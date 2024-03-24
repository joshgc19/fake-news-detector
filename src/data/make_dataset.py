# -*- coding: utf-8 -*-
import re
import string
import pandas as pd
import math


def word_cleaning(text):
    text = text.lower().strip()
    text = re.sub('[.*?]|\w*\d\w*|\n|https?:\\\S+|www\.\S+|<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\W', " ", text)
    return text


def clean_text(text: pd.DataFrame):
    """
    Function that applies a cleaning procedure to every row in a DataFrame
    Args:
        text(pd.DataFrame):

    Returns:

    """
    return text.apply(word_cleaning)


def split_data(matrix: pd.DataFrame, target_column: str, test_ratio: int = 0.25):

    matrix = matrix.sample(frac=1)
    test_matrix_length = math.floor(matrix.shape[0] * test_ratio)
    test_dataset = matrix[:test_matrix_length]
    train_dataset = matrix[test_matrix_length:]
    return train_dataset, test_dataset


def make_dataset(fake_data: pd.DataFrame, true_data: pd.DataFrame):
    # Add target column to both datasets to label them as fake or true news
    fake_data['target'] = 0
    true_data['target'] = 1

    # Concatenate both dataframes into one
    merged_data = pd.concat([fake_data, true_data], axis=0)
    del fake_data, true_data  # Memory optimization

    # Drop not needed columns such as title, subject and date. We are only focusing on content.
    merged_data = merged_data.drop(['title', 'subject', 'date'], axis=1)

    # Clean contents
    merged_data['text'] = clean_text(merged_data['text'])

    # Split dataset into training and testing sets
    return split_data(merged_data, 'target')
