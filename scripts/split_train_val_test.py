import argparse
import csv
import os.path
import random


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


def write_to_csv(data: list[list[str]], filename: str):
    with open(filename, "w", encoding="utf-8") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerows(data)


def main(args: argparse.Namespace):
    data = []
    header = []
    for file_name in args.data:
        with open(file_name, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file, delimiter=";")
            header = next(csv_reader)
            for line in csv_reader:
                data.append(line)

    random.shuffle(data)
    val_size = int(len(data) * args.val_ratio)
    test_size = int(len(data) * args.test_ratio)

    train_offset = val_size + test_size
    partition = {
        "val": data[:val_size],
        "test": data[val_size:train_offset],
        "train": data[train_offset:],
    }

    dirname = os.path.dirname(args.data[-1])
    for group_name, data in partition.items():
        data_with_header = [header] + data
        write_to_csv(
            data_with_header, os.path.join(dirname, f"{group_name}.csv")
        )


if __name__ == "__main__":
    _args = _configure_argparser().parse_args()
    main(_args)
