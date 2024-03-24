
import click
import pandas as pd

from common.files_utils import load_csv_as_dataframe, save_dataframe_as_csv, save_data_to_file
from features.build_features import apply_tf_idf


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True, resolve_path=False))
@click.argument('is_training', type=click.BOOL)
def main(dataset_path, is_training):
    dataset = load_csv_as_dataframe(dataset_path)[:]

    x = dataset['text']
    y = dataset['target']
    del dataset

    features_vectors, word_keys, words = apply_tf_idf(x)
    del x

    if is_training:
        features_df = pd.DataFrame(features_vectors, columns=range(len(words)))
        features_df = pd.concat([features_df, y], axis=1)
        save_data_to_file(str(word_keys), "../models/word_mapping.txt")
        save_dataframe_as_csv(features_df, "../models/training_feature_vectors.csv")
    else:
        return word_keys, features_vectors, y


if __name__ == '__main__':
    main()
