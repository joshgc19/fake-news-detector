# -*- coding: utf-8 -*-
"""
File contains utilitarian function.

This script is used to load the dataset, preprocess it and save it in data/processed/.

The output files are divided into train and test preprocessed and their names are "tran_data.csv" and "test_data.csv".

This file contains the following functions:

    * main - main function of the script
"""
import pandas as pd


def load_csv_as_dataframe(origin_path: str):
    try:
        return pd.read_csv(origin_path)
    except Exception as e:
        raise Exception(f'CSV file at path: {origin_path} could not be loaded. Check path and try again.')


def save_dataframe_as_csv(df: pd.DataFrame, target_path: str):
    try:
        df.to_csv(target_path, index=False)
    except Exception as e:
        raise Exception(f'Could not save dataframe to path: {target_path}. Check path and try again.')


def save_data_to_file(data: str, path: str):
    with open(path, 'w', encoding="utf-16") as f:
        f.write(data)
