# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from ..common.files_utils import load_csv_as_dataframe, save_dataframe_as_csv
from ..common.preprocessing_utils import word_cleaning, split_data


def preprocess_data(fake_data: pd.DataFrame, true_data: pd.DataFrame):
    # Add target column to both datasets to label them as fake or true news
    fake_data['target'] = 0
    true_data['target'] = 1

    # Concatenate both dataframes into one
    merged_data = pd.concat([fake_data, true_data], axis=0)
    del fake_data, true_data  # Memory optimization

    # Drop not needed columns such as title, subject and date. We are only focusing on content.
    merged_data = merged_data.drop(['title', 'subject', 'date'], axis=1)

    # Clean contents
    merged_data['text'] = word_cleaning(merged_data['text'])

    # Split dataset into training and testing sets
    return split_data(merged_data, 'target')


# Click allows to pass arguments from command line, in this example it would be called as python input_filepath=<path>
# output_filepath=<path>
@click.command()
@click.argument('--fake_news_filepath', type=click.Path(exists=True))
@click.argument('--true_news_filepath', type=click.Path(exists=True))
@click.argument('--output_filepath', type=click.Path())
def main(fake_news_filepath, true_news_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Read fake and true news data sets as a panda dataframe
    fake_data = load_csv_as_dataframe(fake_news_filepath)
    true_data = load_csv_as_dataframe(true_news_filepath)

    # Preprocess fake and true news data
    train_data, test_data = preprocess_data(fake_data, true_data)

    # Write preprocessed news data
    save_dataframe_as_csv(train_data, output_filepath + "/train_data.csv")
    save_dataframe_as_csv(test_data, output_filepath + "/test_data.csv")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
