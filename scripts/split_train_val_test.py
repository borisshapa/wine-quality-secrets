import argparse
import os.path

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


def main(args: argparse.Namespace):
    dataframes = []

    for ind, filename in enumerate(args.data):
        dataframe = pd.read_csv(filename, sep=utils.CSV_SEPARATOR)
        dataframe.insert(
            0, utils.WINE_TYPE_COLUMN_NAME, [ind] * dataframe.shape[0]
        )
        dataframes.append(dataframe)

    data = pd.concat(dataframes, ignore_index=True)
    shuffled_data = data.sample(frac=1, ignore_index=True)

    partition = utils.split_into_train_val_test(
        shuffled_data, args.val_ratio, args.test_ratio, args.seed
    )

    dirname = os.path.dirname(args.data[-1])
    for group_name, data in partition.items():
        filename = os.path.join(dirname, f"{group_name}.csv")
        data.to_csv(filename, sep=utils.CSV_SEPARATOR, index=False)


if __name__ == "__main__":
    _args = _configure_argparser().parse_args()
    main(_args)
