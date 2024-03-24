# -*- coding: utf-8 -*-
"""
File contains utilitarian functions used to read and write files using pandas and python methods.

This file contains the following functions:

    * load_csv_as_dataframe - using pandas loads archive as a dataframe
    * save_csv_as_dataframe - using pandas writes a dataframe to an archive
    * save_data_to_file - using python method open writes to a new file
"""

import pickle

import pandas as pd


def load_csv_as_dataframe(origin_path: str):
    """
    Function that loads a csv archive as a dataframe and returns it
    Args:
        origin_path(str): path of the csv archive to be loaded

    Returns:
        pd.DataFrame: loaded dataframe
    """
    try:
        return pd.read_csv(origin_path)
    except Exception as e:
        raise Exception(f'CSV file at path: {origin_path} could not be loaded. Check path and try again.')


def save_dataframe_as_csv(df: pd.DataFrame, target_path: str):
    """
    Function that saves a dataframes as a csv archive
    Args:
        df(pd.DataFrame): dataframe to be saved
        target_path(str): file location including name and extension of the output file

    Returns:
        None
    """
    try:
        df.to_csv(target_path, index=False)
    except Exception as e:
        raise Exception(f'Could not save dataframe to path: {target_path}. Check path and try again.')


def save_data_to_file(data: dict | list, path: str):
    """
    Function that saves a data dictionary or list to a new binary file at the specified path
    Args:
        data(str): data to be saved
        path(str): file location including name and extension of the output file

    Returns:
        None
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_file_as_object(path: str, preferred_type: type):
    """
    Function that loads a binary file as an object
    Args:
        path(str): path of the file to be loaded
        preferred_type(type): the preferred type in which the binary file will be cast to

    Returns:
        (type): the loaded object cast
    """
    with (open(path, 'rb') as f):
        object_var = pickle.load(f)
        return preferred_type(object_var)
