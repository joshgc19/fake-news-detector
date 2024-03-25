# -*- coding: utf-8 -*-
"""
File contains utilitarian functions used to read and write files using pandas and python methods.

This file contains the following functions:

    * load_csv_as_dataframe - using pandas loads archive as a dataframe
    * save_csv_as_dataframe - using pandas writes a dataframe to an archive
    * save_data_to_file - using python method open writes to a new file
    * load_file_as_object - using python method loads archive and casts it to a desired type
    * dump_as_joblib_bin - using joblib to dump an object to a binary file
    * load_joblib_as_object - using joblib to load an object
"""

import pickle
from joblib import load, dump

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
        (dict | list |sklearn.tree.DecisionTreeClassifier): the loaded object cast
    """
    with (open(path, 'rb') as f):
        object_var = pickle.load(f)
        return preferred_type(object_var)


def dump_as_joblib_bin(path: str, data: object):
    """
    Procedure that dumps an object to a binary file
    Args:
        path(str): path in which the binary file will be created, including file name and extension
        data(object): data object to be saved as binary

    Returns:
        None
    """
    with open(path, 'wb') as f:
        dump(data, f)


def load_joblib_as_object(path: str, preferred_type: type):
    """
    Function that loads a binary file as an object and casts it
    Args:
        path(str): path in which the binary file is
        preferred_type(type): type in which the object will be cast to

    Returns:
        type: cast object to the preferred type
    """
    with open(path, 'rb') as f:
        return preferred_type(load(f))
