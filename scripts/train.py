import argparse


def _configure_arg_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--train-data",
        type=str,
        default="data/train.csv",
        help="csv file with data for training.",
    )
    argparser.add_argument(
        "--val-data",
        type=str,
        default="data/val.csv",
        help="csv file with data for validation.",
    )
    argparser.add_argument(
        "--config",
        type=str,
        default=""
    )
    return argparser


def main(args: argparse.Namespace):
    pass


if __name__ == "__main__":
    _args = _configure_arg_parser().parse_args()
    main(_args)
