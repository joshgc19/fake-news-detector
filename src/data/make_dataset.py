# -*- coding: utf-8 -*-
"""
This file contains the functions needed to preprocess the data for the Fake News Recognizer.

This file contains the following functions:

    * word_cleaning - function that's applied to each row of the dataset
    * clean_text - function that applies a cleaning function to all the dataset
    * split_date - own function that shuffles and splits the dataset into training and testing sets for cross-validation
    * make_dataset - function that adds needed columns and applied formatting functions to the dataset
"""

import re
import string
import pandas as pd
import math


def word_cleaning(text):
    """
    Function that applies standardization of the given string for further processing
    Args:
        text(str): the string to be cleaned:

    Returns:
        str: cleaned string
    """
    text = text.lower().strip()
    text = re.sub('[.*?]|\w*\d\w*|\n|https?:\\\S+|www\.\S+|<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\W', " ", text)
    return text


def clean_text(corpus: pd.DataFrame):
    """
    Function that applies a cleaning procedure to every row in a DataFrame
    Args:
        corpus(pd.DataFrame): data source to be cleaned

    Returns:
        pd.DataFrame: cleaned DataFrame
    """
    return corpus.apply(word_cleaning)


def split_data(matrix: pd.DataFrame, test_ratio: int = 0.25):
    """
    Owner function that shuffles and splits the dataset into training and testing with the given ratio
    Args:
        matrix(pd.DataFrame): data source to be treated
        test_ratio(int): proportion of the dataset to be used for testing, defaults to 25%

    Returns:
        (pd.DataFrame, pd.DataFrame): Tuple of training and testing datasets as data frames
    """
    matrix = matrix.sample(frac=1)
    test_matrix_length = math.floor(matrix.shape[0] * test_ratio)
    test_dataset = matrix[:test_matrix_length]
    train_dataset = matrix[test_matrix_length:]
    return train_dataset, test_dataset


def make_dataset(fake_data: pd.DataFrame, true_data: pd.DataFrame):
    """
    Function that performs formatting and splitting of the dataset to be used in the extraction of features
    Args:
        fake_data(pd.DataFrame): Dataset containing raw data about false news
        true_data(pd.DataFrame): Dataset containing raw data about true news

    Returns:
        (pd.DataFrame, pd.DataFrame): Train and test datasets respectively
    """
    # Add target column to both datasets to label them as fake or true news
    fake_data['target'] = 0
    true_data['target'] = 1

    # Concatenate both dataframes into one
    merged_data = pd.concat([fake_data, true_data], axis=0)
    del fake_data, true_data  # Memory optimization

    # Drop not needed columns such as title, subject and date. We are only focusing on content or "text".
    merged_data = merged_data.drop(['title', 'subject', 'date'], axis=1)

    # Clean contents
    merged_data['text'] = clean_text(merged_data['text'])

    # Split dataset into training and testing sets
    return split_data(merged_data)
