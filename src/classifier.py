# -*- coding: utf-8 -*-
"""
Classifier and accuracy checker for Fake News Recognizer.

This script is used to extract the features from the datasets so it can be loaded into a Recognition Model.

The output files are stored in 'data/features.

This file contains the following functions:

    * main - main function of the script
"""

import os
import click
from dotenv import load_dotenv
from scipy.sparse import load_npz
from sklearn.tree import DecisionTreeClassifier

from common.files_utils import load_csv_as_dataframe, dump_as_joblib_bin, load_joblib_as_object
from models.train_model import train_model

load_dotenv()


@click.command()
@click.argument('keep_model', type=click.BOOL)
def main(keep_model=True):
    """

    Args:
        keep_model:

    Returns:

    """
    if os.path.exists(os.getenv('MODEL_OBJ')) and keep_model:
        df = load_joblib_as_object(os.getenv('MODEL_OBJ'), DecisionTreeClassifier)
    else:

        train_vectorized_data = os.getenv('FEATURES_DATA_DIR') + os.getenv('TRAIN_FEATURES_MATRIX')
        x = load_npz(train_vectorized_data)

        train_data_file = os.getenv('PROCESSED_DATA_DIR') + os.getenv('TRAIN_DATA_CSV')
        y = load_csv_as_dataframe(train_data_file)['target']

        df = train_model(x, y)

        del x, y, train_data_file, train_vectorized_data
        dump_as_joblib_bin(os.getenv('MODEL_OBJ'), df)

    x_test = load_npz( os.getenv('FEATURES_DATA_DIR') + os.getenv('TEST_FEATURES_MATRIX'))

    y_test = load_csv_as_dataframe(os.getenv('PROCESSED_DATA_DIR') + os.getenv('TEST_DATA_CSV'))['target']

    predictions = df.predict(x_test)
    score = df.score(x_test, y_test)

    test_dataset_len = len(y_test)
    false_positives = 0
    false_negatives = 0
    truth_positives = 0
    truth_negatives = 0

    for prediction, target in zip(predictions, y_test):

        # Count false and truth negatives and positives
        if prediction == 1 and target == 1:
            truth_positives += 1
        elif prediction == 0 and target == 0:
            truth_negatives += 1
        elif prediction == 1 and target == 0:
            false_positives += 1
        elif prediction == 0 and target == 1:
            false_negatives += 1

    # Prints out general statistics
    print("\n== Statistics ==\n")
    print("Truth positives = ", truth_positives, " observations - ", truth_positives / test_dataset_len * 100, "%")
    print("Truth negatives = ", truth_negatives, " observations - ", truth_negatives / test_dataset_len * 100, "%")
    print("False positives = ", false_positives, " observations - ", false_positives / test_dataset_len * 100, "%")
    print("False negatives = ", false_negatives, " observations - ", false_negatives / test_dataset_len * 100, "%")
    print("Accuracy = ", score*100, "%")


if __name__ == "__main__":
    # This function will only be run when this file is ran directly
    main()
