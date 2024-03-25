# -*- coding: utf-8 -*-
"""
Preprocessing for Fake News Recognizer.

This script is used to load the dataset, preprocess it and save it in data/processed/.

The output files are divided into train and test preprocessed and their names are "tran_data.csv" and "test_data.csv".

This file contains the following functions:

    * main - main function of the script
"""

import os
import click
from dotenv import load_dotenv

from common.files_utils import load_csv_as_dataframe, save_dataframe_as_csv
from data.make_dataset import make_dataset

load_dotenv()


# Click allows to retrieve arguments when calling the file from console such as $ python preprocessing.py <args>
@click.command()
@click.argument('fake_news_filepath', type=click.Path(exists=True, resolve_path=False))
@click.argument('true_news_filepath', type=click.Path(exists=True, resolve_path=False))
def main(fake_news_filepath, true_news_filepath):
    """
    Main function of the script that preprocesses the data from the input files, preprocessed them and saves them.
    Args:
        fake_news_filepath(str): Filepath where the fake news dataset is located.
        true_news_filepath(str): Filepath where the true news dataset is located.

    Returns:
        None
    """
    # Read fake and true news data sets as a panda dataframe
    fake_data = load_csv_as_dataframe(fake_news_filepath)
    true_data = load_csv_as_dataframe(true_news_filepath)

    # Preprocess fake and true news data
    train_data, test_data = make_dataset(fake_data, true_data)

    # Write preprocessed news data file path must contain a / at the end
    save_dataframe_as_csv(train_data, os.getenv('PROCESSED_DATA_DIR') + os.getenv('TRAIN_DATA_CSV'))
    save_dataframe_as_csv(test_data, os.getenv('PROCESSED_DATA_DIR') + os.getenv('TEST_DATA_CSV'))


if __name__ == "__main__":
    # This function will only be run when this file is ran directly
    main()
