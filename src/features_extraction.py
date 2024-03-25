# -*- coding: utf-8 -*-
"""
Feature extraction for Fake News Recognizer.

This script is used to extract the features from the datasets so it can be loaded into a Recognition Model.

The output files are stored in 'data/features.

This file contains the following functions:

    * main - main function of the script
"""

import os
import click
from dotenv import load_dotenv

from common.files_utils import load_csv_as_dataframe, save_data_to_file, load_file_as_object
from features.build_features import apply_tf_idf
import scipy.sparse as sp

load_dotenv()


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True, resolve_path=False))
@click.argument('is_training', type=click.BOOL)
def main(dataset_path: str, is_training: bool):
    """
    Main function of the script that takes the dataset, loads it and applies the TF-IDF algorithm and saves the
    features matrices.
    Args:
        dataset_path(str): path where the csv file is located
        is_training(bool): whether the dataset corresponds to the training dataset or not

    Returns:
        scipy.Sparse | None: features matrix of the test dataset that is going to be used in the model accuracy test
    """
    # Loads a csv as pandas dataframe to be used
    dataset = load_csv_as_dataframe(dataset_path)[:]

    # Extracts the temporal features as x
    x = dataset['text']
    y = dataset['target']
    del dataset  # Memory optimization: Dataframe no longer needed

    # Check whether the current dataset is for training or testing
    if is_training:
        # If it is training the unique words list must be created
        features_matrix, word_keys, words = apply_tf_idf(x)
    else:
        # Loading of words lists and words dict used to check the words taken into account when making the
        # prediction model
        word_keys = load_file_as_object(os.getenv('FEATURES_DATA_DIR') + os.getenv('WORD_MAPPING'), dict)
        words = load_file_as_object(os.getenv('FEATURES_DATA_DIR') + os.getenv('WORDS'), list)
        # Applying the TF-IDF algorithm to the testing dataset
        features_matrix = apply_tf_idf(x, word_keys, words)[0]
    del x  # Memory optimization: Corpus no longer needed

    # Saving the sparse matrix to memory, the sparse matrix used is DOK (Dictionary of Keys) which doesn't support
    # to be saved as npz, but can easily be transformed to COO, format used to save as npz
    sp.save_npz(f'{os.getenv('FEATURES_DATA_DIR') + ('train' if is_training else 'test')}_features_vectors.npz', features_matrix.tocoo())
    if is_training:
        del features_matrix, y  # Features matrix saved and no longer needed
        # Saving the dictionaries and words list needed to create feature matrices of testing datasets
        save_data_to_file(word_keys, os.getenv('FEATURES_DATA_DIR') + os.getenv('WORD_MAPPING'))
        save_data_to_file(words, os.getenv('FEATURES_DATA_DIR') + os.getenv('WORDS'))
    else:
        return features_matrix, y


if __name__ == '__main__':
    # This function will only be run when this file is ran directly
    main()
