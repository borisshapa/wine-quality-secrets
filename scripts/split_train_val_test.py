import argparse
import os.path

import loguru
import pandas as pd

from src import utils


def _configure_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data",
        nargs="+",
        type=str,
        default=["data/winequality-red.csv", "data/winequality-white.csv"],
        help="the csv files to split into train val test.",
    )
    argparser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="The proportion of data that will be used for validation.",
    )
    argparser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="The proportion of data that will be used for test.",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Random seed for deterministic partition.",
    )
    return argparser


def main(data: list[str], val_ratio: float, test_ratio: float, seed: int):
    dataframes = []

    for ind, filename in enumerate(data):
        loguru.logger.info("Reading file {} | wine type: {}", filename, ind)

        df = pd.read_csv(filename, sep=utils.CSV_SEPARATOR)
        df.insert(0, utils.WINE_TYPE_COLUMN_NAME, [ind] * df.shape[0])
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    shuffled_df = df.sample(frac=1, ignore_index=True)

    partition = utils.split_into_train_val_test(
        shuffled_df, val_ratio, test_ratio, seed
    )

    dirname = os.path.dirname(data[-1])
    for group_name, df in partition.items():
        filename = os.path.join(dirname, f"{group_name}.csv")
        loguru.logger.info(
            "Saving {} into {} | dataset size: {}",
            group_name,
            filename,
            len(df.index),
        )
        df.to_csv(filename, sep=utils.CSV_SEPARATOR, index=False)


if __name__ == "__main__":
    args = _configure_argparser().parse_args()
    main(**vars(args))
