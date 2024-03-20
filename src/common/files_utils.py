import pandas as pd


def load_csv_as_dataframe(origin_path: str):
    try:
        return pd.read_csv(origin_path)
    except:
        raise Exception(f'CSV file at path: {origin_path} could not be loaded. Check path and try again.')


def save_dataframe_as_csv(df: pd.DataFrame, target_path: str):
    try:
        df.to_csv(target_path)
    except:
        raise Exception(f'Could not save dataframe to path: {target_path}. Check path and try again.')
